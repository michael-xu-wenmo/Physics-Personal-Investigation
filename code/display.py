import matplotlib.pyplot as plt
import numpy as np
import os


class Display:
    def __init__(self, energies, wavefunctions, directoryName):
        self.energies = energies
        self.wavefunctions = wavefunctions

        # creating directory
        directory = directoryName
        count = 1
        while os.path.exists(directory):
            print(
                f'Directory "{directory}" already exists - Renaming to "{directoryName}({count})"'
            )
            directory = f"{directoryName}({count})"
            count += 1
        os.mkdir(directory)
        self.directory = directory

        with open(f"{self.directory}/.gitignore", "w") as file:
            file.write("*")

    def export(self, figure, name):
        figure.savefig(f"{self.directory}/{name}")

    def show(self):
        plt.show()

    def plot_energy(self, bar_label=False, fig_size=(8, 6)):
        fig = plt.figure(figsize=fig_size)
        eplot = fig.add_subplot()

        eplot.set_xlabel("Modes")
        eplot.set_ylabel("Frequency Hz")

        bars = eplot.bar(np.arange(1, len(self.energies) + 1, 1), self.energies)

        if bar_label:
            eplot.bar_label(bars)

        return fig

    def plot_wavefunction(self, figsize=(8, 6)):
        for i in range(len(self.wavefunctions)):
            fig, ax = plt.subplots(figsize=figsize)
            w = self.wavefunctions[i]
            ax.set_aspect("equal", adjustable="box")
            ax.pcolor(w)
            ax.contour(w, levels=0, colors="black")
            fig.suptitle(f"{i}: {self.energies[i]}")
            yield fig
