import os
import logging.config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logging.config.fileConfig(
    os.path.dirname(os.path.abspath(__file__)) + "/logging_config.ini"
)


class PSO:
    def __init__(
        self,
        cost_function,
        max_vars: list[float],
        min_vars: list[float],
        maximization: bool = True,
        nvars: int = None,
        npop: int = 2,
        interation_limit: int = 2,
        kappa: int = 1,
        phis: list[int] = [2, 2],
        wdamp: int = 1,
        info_display: bool = False,
        output_path: str = "output",
    ) -> None:
        self.cost_function = cost_function
        self.maximization = maximization
        self.nvars = nvars
        self.min_vars = np.array(min_vars)
        self.max_vars = np.array(max_vars)
        self.npop = npop
        self.interation_limit = interation_limit
        self.kappa = kappa
        self.phi1 = phis[0]
        self.phi2 = phis[1]
        self.phi = sum(phis)
        self.wdamp = wdamp
        self.info_display = info_display
        self.chi = (
            2 * kappa / abs(2 - self.phi - (self.phi**2 - 4 * self.phi1) ** (1 / 2))
        ) * wdamp
        self.output_path = output_path

        self.chi1 = self.chi * self.phi1
        self.chi2 = self.chi * self.phi2

        self.max_velocity = 0.2 * (self.max_vars - self.min_vars)
        self.min_velocity = -self.max_velocity

        cols = [f"value{q}" for q in range(nvars)]
        cols.append("cost")
        self.df_results = pd.DataFrame(columns=cols)

    def save_particle_cost_csv(self, data: list) -> None:
        cols = [f"value{q}" for q in range(self.nvars)]
        cols.append("cost")
        df_result = pd.DataFrame(data, columns=cols)
        df_result.to_csv(
            f"{self.output_path}/position_cost_result.csv", float_format="%.2f"
        )

    def draw_scatter_plot(
        self,
        values_list,
        title="PSO - Iterations Display",
        x_label="Iteration",
        y_label="Cost",
    ):
        x_values = []
        y_values = []

        for idx, value in enumerate(values_list):
            x_values.append(idx + 1)
            y_values.append(value)

        # plt.ion()  # Turn on interactive mode

        if self.maximization:
            optmum_y_index = y_values.index(
                max(y_values)
            )  # Find the index of the max y value
        else:
            optmum_y_index = y_values.index(
                min(y_values)
            )  # Find the index of the min y value
        colors = [
            "orange" if i == optmum_y_index else "gray" for i in range(len(x_values))
        ]  # Set color to red for max y value

        plt.scatter(x_values, y_values, marker="o", color=colors)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        # plt.show()
        # plt.pause(0.05)  # Pause to allow time for the plot to be displayed
        plt.savefig(
            f"{self.output_path}/iteration_plot.png", dpi=300, bbox_inches="tight"
        )

    def initialize_global_best(self) -> list:
        global_best = {}

        if self.maximization:
            global_best["cost"] = np.inf * -1
        else:
            global_best["cost"] = np.inf

        global_best["position"] = []

        return global_best

    def initialize_particles(self) -> list:
        particles = []

        for _ in range(self.npop):
            particle_structure = {}
            particle_structure["position"] = np.round(
                np.random.uniform(self.min_vars, self.max_vars, size=self.nvars), 3
            )
            particle_structure["velocity"] = np.zeros(self.nvars)
            particle_structure["cost"] = []
            particle_structure["best"] = {}
            particle_structure["best"]["position"] = np.zeros(self.nvars)
            particle_structure["best"]["cost"] = self.initialize_global_best()["cost"]
            particles.append(particle_structure)

        return particles

    def update_best_position(self, particle: dict, global_best: dict) -> dict:
        # MAX
        if self.maximization:
            if particle["cost"] > particle["best"]["cost"]:
                particle["best"]["position"] = particle["position"]
                particle["best"]["cost"] = particle["cost"]

                if particle["best"]["cost"] > global_best["cost"]:
                    global_best = particle["best"]

        # MIN
        else:
            if particle["cost"] < particle["best"]["cost"]:
                particle["best"]["position"] = particle["position"]
                particle["best"]["cost"] = particle["cost"]

                if particle["best"]["cost"] < global_best["cost"]:
                    global_best = particle["best"]

        return particle, global_best

    def update_velocity(self, particle: dict, global_best: dict) -> list:
        particle["velocity"] = (
            self.chi * particle["velocity"]
            + self.chi1
            * np.random.rand(self.nvars)
            * (particle["best"]["position"] - particle["position"])
            + self.chi2
            * np.random.rand(self.nvars)
            * (global_best["position"] - particle["position"])
        )

        particle["velocity"] = np.maximum(particle["velocity"], self.min_velocity)
        particle["velocity"] = np.minimum(particle["velocity"], self.max_velocity)

        return particle["velocity"]

    def executer(self) -> dict:
        global_best = self.initialize_global_best()
        particle = self.initialize_particles()
        all_data = []
        all_cost = []

        for it in range(self.interation_limit):
            for i in range(self.npop):
                if it != 0:
                    # ----------------------------------- Update Velocity ---------------------------------- #
                    particle[i]["velocity"] = self.update_velocity(
                        particle=particle[i], global_best=global_best
                    )

                    # ----------------------------------- Update Position ---------------------------------- #
                    particle[i]["position"] = (
                        particle[i]["position"] + particle[i]["velocity"]
                    )
                    particle[i]["position"] = np.maximum(
                        particle[i]["position"], self.min_vars
                    )
                    particle[i]["position"] = np.minimum(
                        particle[i]["position"], self.max_vars
                    )
                    pass

                # ------------------------------------- Evaluation ------------------------------------- #
                particle[i]["cost"] = self.cost_function(particle[i]["position"])
                logging.info(
                    f"Iteration {it+1} / Indivudual: {i+1}/{self.npop} / Particle Cost: {particle[i]['cost']:,.3}"
                )

                it_data = list(particle[i]["position"])
                it_data.append(particle[i]["cost"])
                all_data.append(it_data)

                all_cost.append(particle[i]["cost"])
                self.draw_scatter_plot(all_cost)

                # Update particle and global bests
                particle[i], global_best = self.update_best_position(
                    particle=particle[i], global_best=global_best
                )

        self.save_particle_cost_csv(all_data)

        return global_best
