import numpy as np


def superposition(freqs, patterns):
    exist = {}
    for i, f in enumerate(freqs):
        if f in exist.keys():
            exist[f].append(i)
        else:
            exist[f] = [i]

    superposed = []
    sfreq = []

    for paired in exist.keys():
        if len(exist[paired]) < 2:
            continue
        sfreq.append(paired)
        superposed.append(patterns[exist[paired][0]] + patterns[exist[paired][0]].T)

        sfreq.append(paired)
        superposed.append(
            np.rot90(patterns[exist[paired][1]] + patterns[exist[paired][1]].T)
        )

        sfreq.append(paired)
        superposed.append(
            patterns[exist[paired][0]]
            + patterns[exist[paired][0]].T
            + np.rot90(patterns[exist[paired][1]] + patterns[exist[paired][1]].T)
        )

    return np.array(sfreq), np.array(superposed)
