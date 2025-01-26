#!/bin/bash
#SBATCH --job-name=fid_lcm            # Название задачи 
#SBATCH --error=my_array_task-%A-%a.err     # Файл для вывода ошибок 
#SBATCH --output=my_numpy_task-%A-%a.log    # Файл для вывода результатов 
#SBATCH --array=1-3                        # Массив подзадач с номерами от 1 до 20

idx=$SLURM_ARRAY_TASK_ID                    # В переменную idx записывается номер текущей подзадачи
echo "I am array job number" $idx           # Отображение ID подзадачи
python3 compute_fid_lcm_laion.py -i $idx   # Выполнение расчёта с входным файлом input_$idx.dat
