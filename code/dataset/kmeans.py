# Libraries
import cv2
import numpy as np
from typing import List, Tuple

# Constants
OLD_CENTER_INIT = float('inf')
BLUR_KERNEL_SIZE = (5, 5)
THRESHOLD = 0.1
CLUSTERS_NUMBER = 2


def otsu(image: np.ndarray) -> int:
    """
    Applies Otsu's thresholding to an image to find the optimal threshold value.
    
    Args:
        image (numpy.ndarray): Grayscale input image.
        
    Returns:
        int: Optimal threshold value determined by Otsu's method.
    """
    # Apply Gaussian blur to the image
    image = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, 0)
    MN = image.size
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    histogram = histogram.astype(float) / MN  # Normalize histogram

    sigma_max = 0.0
    output = 0

    # Iterate through all possible threshold values to find the one that maximizes sigma
    for k in range(256):
        P1 = np.sum(histogram[:k + 1])  # Probability of class 1
        P2 = np.sum(histogram[k + 1:])  # Probability of class 2

        # Mean of class 1 and class 2
        med_cum_A = np.sum(histogram[:k + 1] * np.arange(k + 1))
        med_cum_B = np.sum(histogram[k + 1:] * np.arange(k + 1, 256))

        # Calculate between-class variance
        sigma = P1 * P2 * (med_cum_A - med_cum_B) ** 2

        # Update if the current variance is greater than the previous maximum
        if sigma > sigma_max:
            sigma_max = sigma
            output = k

    print(output)
    return output


def create_clusters_info(img_input: np.ndarray, clusters_number: int) -> Tuple[List[Tuple], List[List[Tuple]]]:
    """
    Initializes cluster centers and point lists for each cluster.
    
    Args:
        img_input (numpy.ndarray): Input image.
        clusters_number (int): Number of clusters.
        
    Returns:
        tuple: A list of initial cluster centers and a list of points for each cluster.
    """
    clusters_centers = []
    pt_in_clusters = [[] for _ in range(clusters_number)]

    # Random number generator
    random = np.random.default_rng()

    # Randomly select initial cluster centers from the image
    for _ in range(clusters_number):
        center_k_point = (random.integers(0, img_input.shape[1]), random.integers(0, img_input.shape[0]))
        center_pixel = img_input[center_k_point[1], center_k_point[0]]
        clusters_centers.append(tuple(center_pixel))

    return clusters_centers, pt_in_clusters


def compute_color_distance(pixel: Tuple[int, int, int], cluster_pixel: Tuple[int, int, int]) -> float:
    """
    Computes the Euclidean distance between two RGB pixels.
    
    Args:
        pixel (tuple): RGB values of the first pixel.
        cluster_pixel (tuple): RGB values of the cluster center pixel.
        
    Returns:
        float: The distance between the two pixels.
    """
    return np.linalg.norm(np.array(pixel) - np.array(cluster_pixel))


def find_associated_cluster(img_input: np.ndarray, clusters_centers: List[Tuple], pt_in_clusters: List[List[Tuple]]) -> None:
    """
    Assigns each pixel of the image to the closest cluster based on color distance.
    
    Args:
        img_input (numpy.ndarray): Input image.
        clusters_centers (list): List of current cluster centers.
        pt_in_clusters (list): List of points assigned to each cluster.
    """
    for r in range(img_input.shape[0]):
        for c in range(img_input.shape[1]):
            pixel = img_input[r, c]
            distances = [compute_color_distance(pixel, center) for center in clusters_centers]
            closest_cluster_index = np.argmin(distances)
            pt_in_clusters[closest_cluster_index].append((c, r))


def adjust_cluster_centers(img_input: np.ndarray, clusters_centers: List[Tuple], pt_in_clusters: List[List[Tuple]]) -> Tuple[float, List[Tuple]]:
    """
    Adjusts the cluster centers based on the mean color of points assigned to each cluster.
    
    Args:
        img_input (numpy.ndarray): Input image.
        clusters_centers (list): List of current cluster centers.
        pt_in_clusters (list): List of points assigned to each cluster.
        
    Returns:
        tuple: The change in cluster center values and the new cluster centers.
    """
    diff_change = 0
    new_centers = []

    for k, pts in enumerate(pt_in_clusters):
        if pts:
            new_pixel = np.mean([img_input[y, x] for x, y in pts], axis=0)
            diff_change += compute_color_distance(new_pixel, clusters_centers[k])
            new_centers.append(tuple(new_pixel))
        else:
            # Keep old center if no points assigned
            new_centers.append(clusters_centers[k])

    diff_change /= len(clusters_centers)
    print(f"Change in cluster centers: {diff_change}")
    return diff_change, new_centers


def apply_final_cluster_to_image(img_output: np.ndarray, clusters_centers: List[Tuple[int, int, int]], pt_in_clusters: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    Applies the final cluster colors to the output image based on assigned clusters.
    
    Args:
        img_output (numpy.ndarray): Output image to be colored.
        clusters_centers (list): List of final cluster centers.
        pt_in_clusters (list): List of points assigned to each cluster.
        
    Returns:
        numpy.ndarray: Image with colors applied based on clustering.
    """
    for k, pts in enumerate(pt_in_clusters):
        for point in pts:
            pixel_color = clusters_centers[k]
            img_output[point[1], point[0]] = pixel_color

    return img_output


def main() -> None:
    """Main function to perform image segmentation using K-means clustering and Otsu's thresholding.

    This function loads an input image, initializes clusters for K-means, assigns pixels to these clusters, 
    adjusts cluster centers, applies the final clustering results to the image, and then applies Otsu's 
    thresholding to create a binary output. Finally, it displays the original and segmented images.
    """
    # Load the input image in color mode
    img_input = cv2.imread("../datasets/prepared_data/train/A55/5a12f8e3c7ddd91817e041e628ec4317.png", cv2.IMREAD_COLOR)

    if img_input is None:
        print("Error opening image.")
        return

    # Initialize cluster centers and point lists
    clusters_centers, pt_in_clusters = create_clusters_info(img_input, CLUSTERS_NUMBER)

    diff_change = OLD_CENTER_INIT

    # Loop until the change in cluster centers is below the threshold
    while diff_change > THRESHOLD:
        # Clear points in each cluster before reassignment
        pt_in_clusters = [[] for _ in range(CLUSTERS_NUMBER)]

        # Assign pixels to clusters and adjust cluster centers
        find_associated_cluster(img_input, clusters_centers, pt_in_clusters)
        diff_change, clusters_centers = adjust_cluster_centers(img_input, clusters_centers, pt_in_clusters)

    # Copy the input image for the final segmented output
    img_output_knn = img_input.copy()
    img_output_knn = apply_final_cluster_to_image(img_output_knn, clusters_centers, pt_in_clusters)

    # Apply Otsu thresholding to the segmented image
    th = otsu(cv2.cvtColor(img_output_knn, cv2.COLOR_BGR2GRAY))  # Convert to grayscale for Otsu
    _, img_output_knn = cv2.threshold(cv2.cvtColor(img_output_knn, cv2.COLOR_BGR2GRAY), th, 255, cv2.THRESH_BINARY_INV)

    # Display original and segmented images
    cv2.imshow("Original Image", img_input)
    cv2.imshow("Segmentation", img_output_knn)

    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Ensure all windows are closed properly


if __name__ == "__main__":
    main()
