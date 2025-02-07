from decimal import Decimal
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.optimize import minimize, OptimizeResult
from scipy.spatial.distance import cdist
from functools import partial
import matplotlib.pyplot as plt
from typing import Callable, Iterable
import parse
import loopfit as lf
from pathlib import Path

PP = 4 # Precsion for decimal point printing

class SimFilenameParser:
    """
    Allows for embedding metadata (E.g tunable values) into the filename
    of a simulation file.
    """
    def __init__(
        self, prefix: str, param1_delim: str = 'cap',
        param2_delim: str = 'coup', replace_delim: str = '__',
        file_type: str | None = 'ts', prec: int = 5
    ):
        self.prefix = prefix
        self.p1_delim = param1_delim
        self.p2_delim = param2_delim
        self.replace_delim = replace_delim
        self.ftype = file_type
        self.prec = Decimal(f'1e-{prec}')
        if self.ftype is None:
            self.pattern = f'{prefix}_{param1_delim}_{{param_1:f}}_{param2_delim}_{{param_2:f}}'
        else:
            self.pattern = f'{prefix}_{param1_delim}_{{param_1:f}}_{param2_delim}_{{param_2:f}}.{self.ftype}'


    def build_filename(self, param_1: float, param_2: float) -> str:
        """
        Builds a filename for simulation files that embeds metadata. Will replace '.'
        with `replace_delim`. Currently only supports two parameters.
        """
        param_1 = Decimal(param_1).quantize(self.prec)
        param_2 = Decimal(param_2).quantize(self.prec)
        interim = f'{self.prefix}_{self.p1_delim}_{param_1}_{self.p2_delim}_{param_2}'
        interim = interim.replace('.', self.replace_delim)
    
        if interim is None:
            return interim
        return f'{interim}.{self.ftype}'

    def parse_filename(self, fname: str) -> tuple[float, float] | None:
        """
        Parses a filename generated using this instance of the
        parser. Will return None if pattern is not matched.
        """
        # First need to get '.' back in the correct places
        fname = fname.replace(self.replace_delim, '.')
        result = parse.parse(self.pattern, fname)
        if result is None:
            return result
        return tuple([v for v in result.named.values()])

class FitIQ:
    """
    Encapulates the logic for retrieving results from a simulation,
    performing fit on the result, extracting the target value(s).
    Expects a path to the simulation results along with any tunable values
    to be associated with it.

    E.g. (
            ("path_to_my_touchstone_file_1", (0.87, 1)),
            ("path_to_my_touchstone_file_2", (0.90, 1)),
            ...,
            ("path_to_my_touchstone_file_n, (1, 1))
         )
    """
    def __init__(
        self, ts_objects: Iterable[tuple[Path | str, tuple[float]]]
    ) -> None: 
        self.ts_objects = ts_objects
        self._fit_params = ("f0", "qc", "qi", "a", "xa", "q0")
        self.res = None

    def fit(
        self,
        summary_on_fail: bool = False,
        phase0_fix: float = 0,
        phase1_fix: float = 0,
        fit_params: str | tuple[str] = ('f0', 'qc')
    ) -> tuple[tuple[float, float], tuple[float, float]] | None: # E.g. ((cap_fill, coupler_fill), (f0, qc))
        """
        Loads the touchstone files and performes the fit on each
        """
        # Check given fit_param is supported
        if isinstance(fit_params, str):
            fit_params = (fit_params,)
        for param in fit_params:
            if param not in self._fit_params:
                raise ValueError(f'{param} is not supported. Supported options are {self._fit_params}.')
        
        # "Pathify" if path is not a Path
        for path, geom_var in self.ts_objects:
            if isinstance(path, str):
                path = Path(path)
            if path.suffix != ".ts":
                raise RuntimeError("Input touchstone files only.")
            
            # Extract the data from the touchstone file for fitting
            try:
                f, i, q = lf.load_touchstone(path)
            except ValueError as e:
                print(f'{e}. Moving to next item.')
                continue

            # Create a guess for the fit. Fixing phase pre-factors
            guess = lf.guess(f, i ,q, phase0=phase0_fix, phase1=phase1_fix)

            # Perform fit. Notify user on failure to converge and continue
            # to next file.
            try:
                result = lf.fit(f, i, q, **guess)
                if not result['success']:
                    print(f'Fit did not converge for {path.name}.')
                    print('Moving to next item.')
                    if summary_on_fail:
                        print(f'Solver summary below:')
                        print(result['summary'])
                    continue
                if self.res is None:
                    self.res = []
            except Exception as e:
                print(f'Fit Exception: {e}.')
                print('Moving to next item.')
                continue

            # Add results to temp storage
            param_strs = [f'{p} = {result[p]}' for p in fit_params]
            print(f'Result for {path.name}: geometry parameter(s) {geom_var}): {", ".join(param_strs)}')
            self.res.append(((geom_var, tuple([result[p] for p in fit_params]))))
        
        # The list holding return values will stay None until successful fit is found
        if self.res is None:
            return None
        return tuple(self.res)

class OptimizationError(BaseException):
    """
    General optimization error for this object.
    """

class ResOptimizer:
    """
    Performs 2D multi-objective optimization to try and find the
    tunable values (E.g. capacitor fill and coupler length) to
    produce a resonator with two target fit parameters (E.g. f0 and Qc).
    This algorithm assumes that BOTH fit parameters depend on BOTH tunables
    (aka, multi-objective). Should work in the independent case, but may not be
    the most efficient.
    """
    def __init__(
            self, input_coords: NDArray, # n_points x 2, coordinates in the grid; E.g. [[cap_fill0, coup_fill0], [cap_fill0, coup_fill1], ...]
            fit_sols1: NDArray, fit_sols2: NDArray, # n_points x 1 for each, should be sols for the associated coord for each fit param
            normalize: bool = True, 
            input_param1_label: str = 'cap_fill',
            input_param2_label: str = 'coupler_fill',
            fit_param1_label: str = 'f0',
            fit_param2_label: str = 'qc'
    ):
        self.f1_objective: Callable[..., float] | None = None
        self.f2_objective: Callable[..., float] | None = None
        self.fit_sols1: NDArray = fit_sols1
        self.fit_sols2: NDArray = fit_sols2
        self.norm = normalize
        self.fit1_norm = 1.0
        self.fit2_norm = 1.0
        if self.norm:
            self.fit1_norm = fit_sols1.max()
            self.fit2_norm = fit_sols2.max()
            self.fit_sols1 = self.fit_sols1 / self.fit1_norm
            self.fit_sols2 = self.fit_sols2 / self.fit2_norm
        
        self.coords: NDArray = input_coords
        self.bounds: NDArray | None = None
        self.input1_label = input_param1_label
        self.input2_label = input_param2_label
        self.fit1_label = fit_param1_label
        self.fit2_label = fit_param2_label
        self.merged_fit_sols = np.array(list(zip(self.fit_sols1, self.fit_sols2)))

    
    def interpolate(self, plot: bool = False, **kwargs) -> None:
        """
        Performs 2D interpolation using the Clough-Tocher scheme for both fit params.
        Kwargs are passed to the scipy.interpolate CloughTocher2DInterpolator
        class.
        """
        print("Starting interpolation of the first fit parameter space...")
        self.f1_objective = CloughTocher2DInterpolator(
            self.coords,
            self.fit_sols1,
            **kwargs
        )
        print("Finished interpolation of first fit parameter space!\n")
        
        print("Starting interpolation of the second fit parameter space...")
        self.f2_objective = CloughTocher2DInterpolator(
            self.coords,
            self.fit_sols2,
            **kwargs
        )
        print("Finished interpolation of second fit parameter space!\n")

        if self.norm:
            print("NOTE: Actual values are being normalized, plot is showing non-normalized values!!")

        if plot:
            self._plot_2d(
                x1_label=self.input1_label, x2_label=self.input2_label,
                z_label1=self.fit1_label, z_label2=self.fit2_label
            )

    def _optimize(
            self, guess: NDArray, target: NDArray, f0_tol: float, show_message: bool = False
        ) -> OptimizeResult:
        """
        Handles low-level optimization logic. RAISES ON NON-CONVERGENCE. kwargs are passed
        the multi-objective optimization function.
        """
        # Fix target and f0 tolerance if using normalization
        if self.norm:
            target[0] = target[0] / self.fit1_norm
            target[1] = target[1] / self.fit2_norm
            f0_tol = f0_tol / self.fit1_norm
        
        # Define the objective function for Qc
        qc_opt_fn = lambda xs: np.abs(self.f2_objective(*xs) - target[1])

        # Define the constraint function on resonant frequency and constraint object
        # in format necessary for the optimizer
        f0_constraint_fn = lambda xs: f0_tol - np.abs(self.f1_objective(*xs) - target[0])
        constraints = {'type': 'ineq', 'fun': f0_constraint_fn}

        opt_result = minimize(
            fun=qc_opt_fn,
            x0=guess,
            method='SLSQP',
            constraints=constraints,
            bounds=self.bounds,
            options={'maxiter': 10000}
        )
        if opt_result.success:
            return opt_result

        else:
            err_message = f"Optimization for target ({target[0] * self.fit1_norm:.{PP}f}, {target[1] * self.fit2_norm:.{PP}f}) failed"
            if show_message:
                raise OptimizationError(err_message + f': {opt_result.message}')
            else:
                raise OptimizationError(err_message)


    def optimize(
        self, target: NDArray, f0_tol: float, guess: NDArray | None = None,
        show_message: bool = False
    ) -> tuple[NDArray, NDArray]:
        """
        Performs constrained, multi-objective optimization for a given target.
        Returns ([target_fit_val1, target_fit_val2], [optimized_tunable_val1, optimized_tunable_val2]).
        If guess is None, will find an appropriate guess from the input space (I.E. A coordinate
        that has a fit value pair close to the target). WILL RAISE IF OPTIMIZATION IS UNSUCCESFUL.
        """
        if self.bounds is None:
            print("Finding input space bounds to constrain optimization...")
            self.bounds = self._find_bounds()
            print(f"Bounds found: x1: {self.bounds[0]}, x2: {self.bounds[1]}\n")

        print(f"Starting optimization for target: ({target[0] * self.fit1_norm:.{PP}f}, {target[1] * self.fit2_norm:.{PP}f})...")
        if guess is None:
            print("Finding best initial guess(s) from input space...")
            guess = self._find_guess(target=target)

            # If degenerate guesses for a given target, optimize using each guess
            # and choose the one with smallest distance
            if guess.shape[0] > 1:
                print(f'Found multiple degenerate guesses.')
                dlist = []
                for g in guess:
                    print(f'\tOptimizing with guess: ({g[0] * self.fit1_norm:.{PP}f}, {g[1] * self.fit2_norm:.{PP}f})...')
                    opt_res = self._optimize(
                        guess=g, target=target, f0_tol=f0_tol,
                        show_message=show_message,
                    )
                    dlist.append((opt_res.fun, opt_res.x, g))
                    print(f'\tOptimization successful for guess: ({g[0]:.{PP}f}, {g[1]:.{PP}f}).')
                    print(f'\tDistance: {opt_res.fun}\n')
                
                # Sort on distance value
                dlist.sort(key=lambda x: x[0])
                best_guess = dlist[0]
                print(f'Lowest distance {best_guess[0]:.{PP}f} with guess: ({best_guess[2][0] * self.fit1_norm:.{PP}f}, {best_guess[2][1] * self.fit2_norm}:.{PP}f)')
            else:
                print(f"Found guess: ({guess[0, 0] * self.fit1_norm:.{PP}f}, {guess[0, 1] * self.fit2_norm:.{PP}f})\n")
                opt_res = self._optimize(
                    guess=guess.flatten(), target=target, f0_tol=f0_tol,
                    show_message=show_message,
                )
                best_guess = (opt_res.fun, opt_res.x, guess)

            print(f"Optimization successful for target: ({target[0] * self.fit1_norm:.{PP}f}, {target[1] * self.fit2_norm:.{PP}f})")
            print(f"\tInterp soln: ({self.f1_objective(best_guess[1])[0] * self.fit1_norm:.{PP}f}, {self.f2_objective(best_guess[1])[0] * self.fit2_norm:.{PP}})")
            print(f"\tInput params: ({best_guess[1][0]:.{PP}f}, {best_guess[1][1]:.{PP}f})\n\n")
            return tuple(target * np.array((self.fit1_norm, self.fit2_norm))), tuple(best_guess[1])

    def _plot_2d(
            self, x1_label: str, x2_label: str,
            z_label1: str, z_label2: str,
            figsize: tuple[float, float] = (20, 10),
            title: tuple[str, str] | str = 'Objective Interpolation',
            grid_side: int = 100
    ) -> None:
        """
        Plots the results of the two interpolation operations.
        """
        # Unify title API
        if isinstance(title, str):
            title = (f'{title} {z_label1}', f'{title} {z_label2}')
        
        # Find bounds to fix the plot ranges
        if self.bounds is None:
            print("Finding input space bounds for setting plot limits..")
            self.bounds = self._find_bounds()
            print(f"Bounds found: x1: {self.bounds[0]}, x2: {self.bounds[1]}\n\n")
        
        # Create input arrays for objective functions
        x1 = np.linspace(self.bounds[0, 0], self.bounds[0, 1], grid_side)
        x2 = np.linspace(self.bounds[1, 0], self.bounds[1, 1], grid_side)
        X1, X2 = np.meshgrid(x1, x2)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns

        # Plot the first function
        contour1 = axes[0].contourf(X1, X2, self.f1_objective(X1, X2) * self.fit1_norm, cmap='viridis')
        fig.colorbar(contour1, ax=axes[0], label=z_label1)
        axes[0].set_title(title[0])
        axes[0].set_xlabel(x1_label)
        axes[0].set_ylabel(x2_label)

        # Plot the second function
        contour2 = axes[1].contourf(X1, X2, self.f2_objective(X1, X2) * self.fit2_norm, cmap='viridis')
        fig.colorbar(contour2, ax=axes[1], label=z_label2)
        axes[1].set_title(title[1])
        axes[1].set_xlabel(x1_label)
        axes[1].set_ylabel(x2_label)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def _find_bounds(self) -> NDArray:
        """
        Finds bounds in the input space to constrain the optimizer due to the use of finite
        interpolators.
        """
        x1_bounds = (self.coords[:, 0].min(), self.coords[:, 0].max())
        x2_bounds = (self.coords[:, 1].min(), self.coords[:, 1].max())
        return np.array([x1_bounds, x2_bounds])

        
    def _find_guess(self, target: NDArray) -> NDArray:
        """
        Finds an initial optmization guess from the given discrete output space. This isn't
        guaranteed to be optimal since the output space can be complex. Uses Euclidean
        distance between target and a known point from the discrete output space as objective
        function.
        """
        target = np.array([target, target]) # necessary for the matrix-vec equation for euclidean distance
        dists = cdist(self.merged_fit_sols, target, metric='euclidean')[:, 0] # Cols are duplicates, only need one.
        min_dist = dists.min()
        min_indices = np.where(dists == min_dist)

        # Return all possible "best guesses" using Euclidean distance as FoM
        return self.merged_fit_sols[min_indices]
    
        