# ieos_tf

Создание и обучение простой модели на tensorflow 2.x.
____
Для обучения модели настройте конфиг (train_cfg.json), далее запустите скрипт:

`python fit.py -c train_cfg.json`

Для заморозки графа используте freeze_h5.py:

`python freeze_h5.py -i in_model.h5 -o freeze_graph.pb`
