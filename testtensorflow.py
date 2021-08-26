import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

x = [1, 2, 3, 4, 5]
y = [1, 5, 4, 7, 4]

sns.lineplot(x, y)
plt.show()
