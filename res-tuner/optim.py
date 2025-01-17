import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from functools import partial
import matplotlib.pyplot as plt
from typing import Callable

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
            input_param1_label: str = 'cap_fill',
            input_param2_label: str = 'coupler_fill',
            fit_param1_label: str = 'f0',
            fit_param2_label: str = 'qc'
    ):
        self.f1_objective: Callable[..., float] | None = None
        self.f2_objective: Callable[..., float] | None = None
        self.fit_sols1: NDArray = fit_sols1
        self.fit_sols2: NDArray = fit_sols2
        self.coords: NDArray = input_coords
        self.bounds: NDArray | None = None
        self.input1_label = input_param1_label
        self.input2_label = input_param2_label
        self.fit1_label = fit_param1_label
        self.fit2_label = fit_param2_label
        self.merged_fit_sols = np.array(list(zip(fit_sols1, fit_sols2)))
    
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
        print("Finished interpolation of first fit parameter space!")
        
        print("Starting interpolation of the second fit parameter space...")
        self.f2_objective = CloughTocher2DInterpolator(
            self.coords,
            self.fit_sols2,
            **kwargs
        )
        print("Finished interpolation of second fit parameter space!")

        if plot:
            self._plot_2d(
                x1_label=self.input1_label, x2_label=self.input2_label,
                z_label1=self.fit1_label, z_label2=self.fit2_label
            )

    def optimize(
        self, target: NDArray, guess: NDArray | None = None,
        show_message: bool = False
    ) -> tuple[NDArray, NDArray]:
        """
        Performs constrained, multi-objective optimization for a given target.
        Returns ([target_fit_val1, target_fit_val2], [optimized_tunable_val1, optimized_tunable_val2]).
        If guess is None, will find an appropriate guess from the input space (I.E. A coordinate
        that has a fit value pair close to the target). WILL RAISE IF OPTIMIZATION IS UNSUCCESFUL.
        """
        if self.bounds is None:
            print("Finding input space bounds for constrained optimization...")
            self.bounds = self._find_bounds()
            print(f"Bounds found: {self.bounds}")

        if guess is None:
            print("Finding best initial guess from input space...")
            guess = self._find_guess(target=target)
            print(f"Found guess: ({guess[0], guess[1]})!")

        print(f"Starting optimization for target: ({target[0], target[1]})...")
        opt_result = minimize(
            fun = partial(self._multi_objective_dist_function, y1=target[0], y2=target[1]),
            x0 = guess,
            method = 'L-BFGS-B',
            bounds = self.bounds
        )

        if opt_result.success:
            print(f"Optimization successful for target({target[0], target[1]}):")
            print(f"\tInterp soln: ({self.f1_objective(opt_result.x), self.f2_objective(opt_result.x)})")
            print(f"\tInput params: ({opt_result.x[0], opt_result.x[1]})")
            return (target, opt_result.x)
        else:
            err_message = f"Optimization for target ({target[0], target[1]}) failed"
            if show_message:
                raise OptimizationError(err_message + f': {opt_result.message}')
            else:
                raise OptimizationError(err_message)
              

    def _multi_objective_dist_function(self, xs: NDArray, y1: float, y2: float) -> float:
        """
        The multi-objective function that will be optimized over. This
        function minimizes the SQUARE of the Euclidean distance between the target fit value
        pair and a point on the Pareto optimization front. Raises if interpolation
        has not been performed.
        """
        if self.f1_objective is None or self.f2_objective is None:
            raise OptimizationError("Interpolation is incomplete. Run `interpolate()` before optimizing.")
        
        return np.square(np.square(self.f1_objective((xs[0], xs[1])) - y1) + np.square(self.f2_objective((xs[0], xs[1])) - y2))
        
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
            self.bounds = self._find_bounds()
        
        # Create input arrays for objective functions
        x1 = np.linspace(self.bounds[0, 0], self.bounds[0, 1], grid_side)
        x2 = np.linspace(self.bounds[1, 0], self.bounds[1, 1], grid_side)
        X1, X2 = np.meshgrid(x1, x2)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns

        # Plot the first function
        contour1 = axes[0].contourf(X1, X2, self.f1_objective(X1, X2), cmap='viridis')
        fig.colorbar(contour1, ax=axes[0], label=z_label1)
        axes[0].set_title(title[0])
        axes[0].set_xlabel(x1_label)
        axes[0].set_ylabel(x2_label)

        # Plot the second function
        contour2 = axes[1].contourf(X1, X2, self.f2_objective(X1, X2), cmap='viridis')
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
        distance between target and a known point from the discrete output space.
        """
        target = np.array([target, target]) # necessary for the matrix-vec equation for euclidean distance
        ds = cdist(self.merged_fit_sols, target, metric='euclidean')
        return self.merged_fit_sols[np.argmin(ds[:, 0])]
    
        