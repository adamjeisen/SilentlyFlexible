from absl import flags
from absl import app
import importlib
import logging
import os
import buschman_simulation
from utils import save

FLAGS = flags.FLAGS
flags.DEFINE_string('config', '',
                    'Module name of sweep config to use.')
flags.DEFINE_string('log_directory', '', 'Log directory.')


def main(_):
    print(FLAGS.config)
    print(FLAGS.log_dir)
    log_directory = FLAGS.log_directory
    config_module = importlib.import_module(FLAGS.config)
    config = config_module.get_config()

    def _log(log_filename, thing_to_log):
        f_name = os.path.join(log_directory, log_filename)
        logging.info('In file {} will be written:'.format(log_filename))
        logging.info(thing_to_log)
        f_name_open = open(f_name, 'w+')
        f_name_open.write(thing_to_log)

    if config['type'] == 'buschman':
        sim = buschman_simulation.Simulation(**config['simulation_kwargs'])
        sim.reset()
        run_results = sim.run()
        fpath = f'{log_directory}/{sim.get_fpath()}'
        save(run_results=run_results, simulation=sim, fpath=fpath)
    else:
        pass
    _log('config', str(config))


if __name__ == '__main__':
    app.run(main)
