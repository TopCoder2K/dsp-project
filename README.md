Этот репозиторий предназначен для проекта по курсу "Цифровая обработка сигналов".

# Подготовка

Для работы с репозиторием нужно его склонировать:

```
git clone https://github.com/TopCoder2K/dsp-project.git
```

А после установить зависимости:

```
poetry install --without dev
```

Если не стоит poetry, то нужно
[установить](https://python-poetry.org/docs/#installing-with-the-official-installer).

# Модуль 1

Сначала нужно получить АЧХ колонок, потом импульсный отклик выбранного помещения, потом
свернуть тестовый сигнал с полученным импульсным откликом и сравнить результат с реальным.

## Подготовка данных

Свип и розовый шум было решено качать с audiocheck
([ссылка](https://www.audiocheck.net/testtones_highdefinitionaudio.php)), так как там не
требуется регистрация. Но за это платим уменьшением частоты дискретизации в 2 раза. Для
этого нужно скачать 96-ти килогерцовые свип и розовый шум и поместить их в папку
`data/module1`, а после запустить

```
poetry run python3 data/module1/downsample.py
```

Тестовый файл `gt.wav` также нужно положить в папку `data/module1`.

Все записи на диктофон (использовалось стандартное приложение из Andoid 13, а в качестве
микрофона --- микрофон телефона Realme GT Neo 3T), которые я делал для выполнения этого
модуля, лежат в [этой](https://disk.yandex.ru/d/D79WLiQKnvH_wg) папке на Яндекс Диске.
Оттуда их можно и нужно скачать и расположить в папке `data`. Папка содержит следующие
файлы:

- `recorded_sweep_cropped.wav` --- воспроизведение свипа через ноутбук (если верить
  Ubuntu, то частота дискретизации записей тут и далее равна 48000)
- `recorded_corrected_pink_noise_cropped.wav` --- воспроизведение подправленного розового
  шума через ноутбук
- `recorded_gt_cropped.wav` --- воспроизведение тестового сигнала через ноутбук

## Получение импульсного отклика и преобразование тестового сигнала

### Запись свипа

Чтобы минимизировать влияние отражений и фоновых звуков, запись велась в шкафу,
заполненном вещами. Здесь есть две оговорки:

- так как держать ноутбук и телефон вместе неудобно, я положил их на картонную коробку,
  что, с одной стороны, избавило от звуков трения об одежду и моего дыхания, но, с другой
  стороны, могло добавить отражений от этой коробки,
- так как у ноутбука 2 колонки, я открыл дверь шкафа, чтобы не было отражений от неё, а
  телефон положил к дальней от двери колонке.

Запись получилась покороче свипера (примерно на 0.3 секунды). Вероятно, дело в том, что
совсем низкие частоты настолько плохо воспроизводятся ноутбуком, что были заглушены звуком
нажатия на кнопку "Space" (для воспроизведения) и поэтому были обрезаны мною (я
ориентировался на "тишину", следующую сразу за звуком нажатия на кнопку). Из-за такого
различия после загрузки файла в коде я добавил нули в начало, чтобы выровнять длины
(разница в длинах --- примерно 10.000). Ниже представлены спектрограммы оригинального
свипа и записанного: ![afc.png](data/module1/afc.png) (Также в папке `data/module1` можно
посмотреть этот график с учётом разбиения на бины).

### Симуляция реверберации

В качестве помещения я решил выбрать ванную одной из квартир 12-го общежития, так как там
особенно хорошо слышна реверберация. Для получения подправленных розового шума и тестового
сигнала нужно запустить:

```
poetry run python3 module1.py save_corrected_noise_and_gt --bins_cnt [value]
```

(Число бинов можно задать любым натуральным числом, но эксперименты проводились для
`--bins_cnt 32` бинов.)

Так как подправленный тестовый файл `32_bins_afc_corrected_test_48kHz.wav` получился
слишком подправленным, я не увидел смысла проводить с ним эксперименты: колонки явно
меньше портят звук, поэтому записал воспроизведение оригинального `gt.wav`, который лежит
на Яндекс Диске под именем `recorded_gt_cropped.wav`. Более того, если запустить команду
выше с `--bins_cnt 128`, то коррекция станет лишь немного получше. Похоже, что дело не в
плохой коррекции, а в чём-то ещё.

Получение импульсного отклика и предсказанного результата воспроизведения в ванной
делается через команду:

```
poetry run python3 module1.py get_impulse_responce_and_test
```

## Анализ результатов

Если сравнивать `gt.wav`, `recorded_gt_corrected.wav` и `predicted_gt.wav` между собой, то
такое ощущение, что `predicted_gt.wav` --- это просто `gt.wav`, сделанный потише. Никакой
реверберации там неслышно. Я решил посмотреть на получающееся значение `impulse_responce`,
и увидел, что там всего лишь одно число... Что ж, тут остаётся только довериться `scipy`,
так как ошибок я не вижу. Из улучшений можно предложить следующие:

- использовать профессиональный микрофон, так как у телефона, возможно, плоховат (хотя в
  повседневной жизни мне казалось, что он достаточно хороший),
- использовать какое-то специальное ПО для записи, потому что даже в дефолтном режиме
  приложение "Диктофон" делает какую-то обработку записываемого звука.

# Модуль 2

Нужно посчитать разные метрики на выбранных примерах.

## Подготовка данных и подсчёт SNR, SDR, SI-SDR, PESQ

Так как свёртка полученного импульсного отклика с исходным сигналом едва ли даёт какую-то
реверберацию, я решил выбрать "Интересный варинат" (да, да, именно так). Для этого нужно
скачать [шум метро](https://freesound.org/people/15GPanskaHladikova_Danuse/sounds/461143/)
и расположить его в папке `data/module2`. В качестве чистого голоса использовался файл
`gt.wav` из первого модуля.

Подумаем, как написать mixer. На вход ему подаются сигналы
$A_{\text{original}}, A_{\text{noise}}$ и значение $`SNR^*`$. На выходе должен получиться
сигнал $A_{\text{signal}} = A_{\text{original}} + kA_{\text{noise}}$, для которого верно
$`SNR^* = SNR(A_{\text{signal}}, A_{\text{noise}})`$. Выведем формулу для $k$:

```math
SNR^* = 10\log_{10}\frac{\langle A_{\text{original}}^2\rangle}{\langle (kA_{\text{noise}})^2\rangle}
\Longleftrightarrow \frac{\langle A_{\text{original}}^2\rangle}{k^2\langle A_{\text{noise}}^2\rangle}
= 10^{\frac{SNR^*}{10}} \Longleftrightarrow\\
\Longleftrightarrow k = \pm\sqrt{\frac{\langle A_{\text{original}}^2\rangle}{10^{\frac{SNR^*}{10}}\langle A_{\text{noise}}^2\rangle}}.
```

Для определённости я брал значение $k > 0$. Для получения SNR, SDR, SI-SDR, PESQ при
конкретном значении $`SNR^*`$ достаточно запустить

```
poetry run python3 module2.py calc_metrics_for_given_snr --snr_db [value]
```

(Я добавил **вычисление** метрики SNR для проверки корректности генерации смеси.)

## Подсчёт NISQA и DNSMOS, таблица с результатами

Для получения [NISQA](https://github.com/gabrielmittag/NISQA) и
[DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) нужно пройтись по
инструкциям с соответствующих репозиториев. Из-за нехватки времени я не стал встраивать
код репозиториев в свой код, а просто склонировал репозитории в отдельные папки и запустил
модели из них на `[snr_in_dB]_dB_mixture.wav` файлах.

Ниже я собрал метрики для всех запусков в таблицу:
|       Файл        |   SNR   |   SDR   |  SI-SDR |   PESQ  | NISQA, mos_pred | NISQA, noi_pred | NISQA, dis_pred | NISQA, col_pred | NISQA, loud_pred |  DNSMOS |   MOS   |
|:-----------------:|:-------:|:-------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|:----------------:|:-------:|:-------:|
| -5_dB_mixture.wav | -5.0000 | -5.0878 | -5.1124 |  1.0275 |      0.6341     |      1.3234     |      2.8905     |      1.5201     |      1.4479      |  2.1469 |    2    |
|  0_dB_mixture.wav |  0.0000 | -0.0512 | -0.0629 |  1.0422 |      0.8920     |      1.2968     |      3.8234     |      2.8119     |      2.2788      |  2.2924 |    3    |
|  5_dB_mixture.wav |  5.0000 |  4.9724 |  4.9648 |  1.0826 |      1.6883     |      1.2879     |      4.4103     |      3.7602     |      2.8792      |  2.5815 |    4    |
| 10_dB_mixture.wav | 10.0000 |  9.9867 |  9.9803 |  1.1862 |      2.5234     |      1.4233     |      4.4173     |      3.9947     |      3.5467      |  3.2040 |    4    |

## Анализ результатов

Можно сделать следующие выводы:

0. Смешивание реализовано правильно, так как посчитанное значение SNR полностью совпадает
   с точностью до $10^{-4}$ (на самом деле даже до $10^{-6}$);
1. SNR, SDR, SI-SDR довольно сильно коррелируют;
2. Все метрики "отмечают", что с ростом SNR качество становится лучше, однако
   перцептуальные метрики растут заметно медленнее аналитических;
3. PESQ как будто практически не различает эти 4 примера, т.е. она наименее скоррелирована
   со всеми остальными.

# Модуль 3 (TBD)

# Модуль 4 (TBD)
