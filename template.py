import os
import cv2
import numpy as np
from scipy.ndimage import maximum_filter


def find_templ(img, img_tpl):
    # розмір шаблону
    h, w = img_tpl.shape

    a = 0.7  # коефіцієнт схожості, 0 - всі, 1 - точне співпадіння

    # будуємо карту співпадінь зі шаблоном
    match_map = cv2.matchTemplate(img, img_tpl, cv2.TM_CCOEFF_NORMED)

    # значення карти для області максимально близької до шаблону
    max_match_map = np.max(match_map)
    print(max_match_map)
    if(max_match_map <= a):  # співпадінь не знайдено
        return []

    # відрізаємо карту по порогу
    match_map = (match_map >= max_match_map * a) * match_map

    # виділяємо на карті локальні максимуми
    match_map_max = maximum_filter(match_map, size=min(w, h))

    # області, які найбільш близькі до шаблону
    match_map = np.where((match_map == match_map_max), match_map, 0)

    # координати локальних максимумів
    ii = np.nonzero(match_map)
    rr = tuple(zip(*ii))

    res = [[c[1], c[0], w, h] for c in rr]

    return res


# будуємо рамки для співпадінь
def draw_frames(img, coord):
    res = img.copy()
    for c in coord:
        top_left = (c[0], c[1])
        bottom_right = (c[0] + c[2], c[1] + c[3])
        cv2.rectangle(res, top_left, bottom_right, color=(0, 0, 255), thickness=5)
    return res


def main():
    f = "data/juice.jpg"
    bu = "data/juices/"

    templ = [os.path.join(bu, b) for b in os.listdir(bu) if os.path.isfile(os.path.join(bu, b))]
    print(templ)

    # 1 - перетворює зображення на ЧБ.
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    # 2 - знаходить контури на зображенні (canny)
    edges = cv2.Canny(img, 50, 200)
    cv2.imwrite("results/edge.jpg", edges)

    for t in templ:
        print(t)
        img_tpl = cv2.imread(t, cv2.IMREAD_GRAYSCALE)
        coord = find_templ(img, img_tpl)
        img_res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_res = draw_frames(img_res, coord)
        tn = os.path.splitext(os.path.basename(t))[0]
        cv2.imwrite("results/matches/res-%s.jpg" % tn, img_res)
        for c in coord:
            print(c)
        print(len(coord))
        print("______________")


if __name__ == "__main__":
    print("OpenCV ", cv2.__version__)
    main()
