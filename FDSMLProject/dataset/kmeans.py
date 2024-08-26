import cv2
import numpy as np
import math

OLD_CENTER = float('inf')

class Center:
    def __init__(self, blueValue, greenValue, redValue):
        self.blueValue = blueValue
        self.greenValue = greenValue
        self.redValue = redValue


def otsu(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    sigma_max = 0.0
    MN = image.shape[0] * image.shape[1]
    histogram = np.zeros(256, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] += 1

    histogram /= MN
    output = 0

    for k in range(256):
        P1 = np.sum(histogram[:k + 1])
        P2 = np.sum(histogram[k + 1:])

        med_cum_A = np.sum(histogram[:k + 1] * np.arange(0, k + 1))
        med_cum_B = np.sum(histogram[k + 1:] * np.arange(k + 1, 256))

        sigma = P1 * P2 * (med_cum_A - med_cum_B) ** 2

        if sigma > sigma_max:
            sigma_max = sigma
            output = k

    print(output)
    return output


def create_clusters_info(img_input, clusters_number):
    clusters_centers = []
    pt_in_clusters = [[] for _ in range(clusters_number)]
    random = np.random.default_rng()

    for _ in range(clusters_number):
        center_k_point = (random.integers(0, img_input.shape[1]), random.integers(0, img_input.shape[0]))
        center_pixel = img_input[center_k_point[1], center_k_point[0]]
        center_k = tuple(center_pixel)
        clusters_centers.append(center_k)

    return clusters_centers, pt_in_clusters


def compute_color_distance(pixel, cluster_pixel):
    diff_blue = int(pixel[0]) - int(cluster_pixel[0])
    diff_green = int(pixel[1]) - int(cluster_pixel[1])
    diff_red = int(pixel[2]) - int(cluster_pixel[2])
    distance = math.sqrt(diff_blue ** 2 + diff_green ** 2 + diff_red ** 2)
    return distance


def find_associated_cluster(img_input, clusters_number, clusters_centers, pt_in_clusters):
    for r in range(img_input.shape[0]):
        for c in range(img_input.shape[1]):
            min_distance = float('inf')
            closest_cluster_index = 0
            pixel = img_input[r, c]

            for k in range(clusters_number):
                cluster_pixel = clusters_centers[k]
                distance = compute_color_distance(pixel, cluster_pixel)

                if distance < min_distance:
                    min_distance = distance
                    closest_cluster_index = k

            pt_in_clusters[closest_cluster_index].append((c, r))


def adjust_cluster_centers(img_input, clusters_number, clusters_centers, pt_in_clusters,  new_center):
    diff_change = 0

    for k in range(clusters_number):
        pt_in_cluster = pt_in_clusters[k]
        new_blue, new_green, new_red = 0, 0, 0

        for point in pt_in_cluster:
            pixel = img_input[point[1], point[0]]
            new_blue += pixel[0]
            new_green += pixel[1]
            new_red += pixel[2]

        if len(pt_in_cluster) > 0:
            new_blue /= len(pt_in_cluster)
            new_green /= len(pt_in_cluster)
            new_red /= len(pt_in_cluster)

        new_pixel = (new_blue, new_green, new_red)
        new_center += compute_color_distance(new_pixel, clusters_centers[k])
        clusters_centers[k] = new_pixel
    global OLD_CENTER
    new_center /= clusters_number
    diff_change = abs(OLD_CENTER - new_center)
    print(f"diffChange is: {diff_change}")
    OLD_CENTER = new_center

    return diff_change


def apply_final_cluster_to_image(img_output, clusters_number, pt_in_clusters, clusters_centers):
    for k in range(clusters_number):
        pt_in_cluster = pt_in_clusters[k]

        for point in pt_in_cluster:
            pixel_color = clusters_centers[k]
            img_output[point[1], point[0]] = [pixel_color[0], pixel_color[1], pixel_color[2]]

    return img_output


def main():
    img_input = cv2.imread("../datasets/prepared_data/train/A55/5a12f8e3c7ddd91817e041e628ec4317.png", cv2.IMREAD_COLOR)

    if img_input is None:
        print("Error opening image.")
        return

    clusters_number = 2 # Set the number of clusters here

    clusters_centers, pt_in_clusters = create_clusters_info(img_input, clusters_number)

    threshold = 0.1

    new_center = 0
    diff_change = OLD_CENTER - new_center

    while diff_change > threshold:
        new_center = 0

        for k in range(clusters_number):
            pt_in_clusters[k].clear()

        find_associated_cluster(img_input, clusters_number, clusters_centers, pt_in_clusters)
        diff_change = adjust_cluster_centers(img_input, clusters_number, clusters_centers, pt_in_clusters,
                                             new_center)

    img_output_knn = img_input.copy()
    img_output_knn = apply_final_cluster_to_image(img_output_knn, clusters_number, pt_in_clusters, clusters_centers)

    th = otsu(img_output_knn)
    _, img_output_knn = cv2.threshold(img_output_knn, th, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("Immagine originale", img_input)
    cv2.imshow("Segmentation", img_output_knn)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
