from pprint import pprint
from collections import OrderedDict


class Config:
    def __init__(self):
        return

    def load_config(self, config_path):
        print('Config:', config_path)
        with open(config_path, 'r') as f:
            config_list = f.readlines()

        # all, spec, class, seg
        self.config_type_dict = {}

        for each in config_list:
            if len(each) > 4 and each[4] == '#':
                tmp_flag = each.split(' ')[1]
                self.config_type_dict[tmp_flag] = {}

            if each[0] in ['#', '*']:
                continue
            elif each == '\n':
                continue
            else:
                kv = each.strip().split('=')
                k = kv[0].strip()
                v = eval(kv[1].strip())
                self.config_type_dict[tmp_flag][k] = v

        config_type_dict_keys = list(self.config_type_dict.keys())
        for config_type_dict_key in config_type_dict_keys:
            self.__dict__.update(self.config_type_dict[config_type_dict_key])

    # spec config for task
    def _spec(self, kwargs):
        print_dict = OrderedDict()
        state_dict = self._state_dict()

        for k, v in kwargs.items():
            if k not in state_dict:
                print('%s is not in config dict, plz check' % k)
            setattr(self, k, v)
            if isinstance(v, list) and len(v) > 10:
                continue
            else:
                print_dict[k] = v

        state_dict = self._state_dict()
        state_dict.pop('config_type_dict')
        state_dict_keys = sorted(list(state_dict.keys()))
        save_config_save = OrderedDict()
        for key in state_dict_keys:
            save_config_save[key] = state_dict[key]

        if state_dict['mode'] == 'train':
            print('======user config========')
            pprint(print_dict)
            print('==========end============')
            return save_config_save
        else:
            return

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in self.__dict__.items()
                if not k.startswith('_')}


opt = Config()
