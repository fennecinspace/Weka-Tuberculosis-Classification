def test(self, weka, algo_params, bagging_params):
    command = self.get_main_command(weka)

    if weka.bagging_enabled:
        final_command = "{} {} -t '{}' -T '{}' -W {} -- {}".format(command, bagging_params, self.data_path, self.cross_path, weka.weka_class, algo_params)
    else:
        final_command = "{} {} -t '{}' -T '{}'".format(command, algo_params, self.data_path, self.cross_path)

    try:
        returned_raw = sp.check_output(['bash', '-c', final_command], stderr = sp.STDOUT)
        mesure, returned, mesures_cols, mesures_vals = self.get_weka_learning_result(returned_raw, DEFAULT_MESURE, cross = True)
        best_cross_val = '\nCross Validation => {} {}\n'.format(DEFAULT_MESURE, mesure)
        best_cross_val += 'Mesures :\n'
        for col, val in list(zip(mesures_cols, mesures_vals)):
            best_cross_val += '{} : {}\n'.format(col, val)
        return best_cross_val, returned

    except Exception as e:
        lg.exception(e)
        return '', ''

