import matplotlib.pyplot as plt


def plot_r_sens(run_results, **kwargs):
    _plot_sens(key='r_sens', run_results=run_results, **kwargs)


def plot_s_sens(run_results, **kwargs):
    _plot_sens(key='s_sens', run_results=run_results, **kwargs)


def plot_p_sens(run_results, **kwargs):
    _plot_sens(key='p_sens', run_results=run_results, **kwargs)


def plot_r_rand(run_results, **kwargs):
    _plot_rand(key='r_rand', run_results=run_results, **kwargs)


def plot_s_rand(run_results, **kwargs):
    _plot_rand(key='s_rand', run_results=run_results, **kwargs)


def plot_p_rand(run_results, **kwargs):
    _plot_rand(key='p_rand', run_results=run_results, **kwargs)


def plot_s_ext(run_results, **kwargs):
    _plot_sens(key='s_ext', run_results=run_results, **kwargs)


def _plot_rand(key, run_results):
    plt.imshow(run_results[key].T)
    plt.colorbar()
    plt.show()


def _plot_sens(key, run_results, sens_idx=0, title=''):
    plt.imshow(run_results[key][:, sens_idx, :].T)
    plt.title(title)
    plt.colorbar()
    plt.show()
