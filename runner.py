from hough.task_1_homework_solution import main as hough_main


if __name__ == '__main__':
    task_n = int(input('Which task to run: '))
    if task_n == 1:
        hough_main('./images/marker_cut_rgb_512.png',
                   './hough/output/homework_solution.jpg')
    else:
        print('')