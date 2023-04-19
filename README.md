# Thai Tran EE-399-HW-2 
## Abstract
For this project, agiven a file containing 39 different images of faces with 65 lighting sceese. Each of these image have been downsampled to 32 by 32 pixels along with being converted to grayscales. Using these images, the goal of this project is to apply different kinds of linear equations to gather data for correlation matrixes. 
## Section I Introduction and Overview
The main goal of this project is to understand the data set and its structure as well as to practice linear algebra to understand its correlation to machine learning. To obtain this goal, the project contains seven different task
### Task A)
Compute a 100 by 100 correlation matrix by computing the dot projects for the first 100 images within the matrix X. 
### Task B)
Using the correlation matrix computedd in the previous task, determine the two most correlated and the the two most uncorrelated image. 
### Task C)
repeating the process from task (a), but instead of computing a 100x100 correlation matrix,  compute a 10x10 correlation matrix between images. Compute the dot product (correlation) between the images with indices [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]. Then plot the correlation matrix between these images. 
### Task D)
creating a matrix Y by multiplying X with its transpose $$Y=(XX^T)$$
We will then find the first six eigenvectors with the largest magnitude eigenvalue of this matrix Y. This task will help us identify the most important directions in the data that capture the most variation.
### Task E)
Perform Singular Value Decomposition (SVD) on the matrix X. Find the first six principal component directions of the data to identify the most important directions in the data that capture the most variation.
### Task F)
Compare the first eigenvector v1 from task (d) with the first SVD mode u1 from task (e) and compute the norm of the difference of their absolute values. This will to understand how similar or different the two methods are in identifying the most important directions in the data.
### Task G)
Compute the percentage of variance captured by each of the first 6 SVD modes. 
## Section II Theoretical Background
For this project, we used many different linear algebra and statics method. Defined below are the methods we used.
### Matrix Correlation
Matrix correlation is a statistical measure that quantifies the similarity or relationship between two matrices. It is often used to analyze the linear association or similarity between two sets of data that are arranged in matrix form. Matrix correlation is typically computed using matrix-based statistical techniques, such as covariance or correlation coefficients, which are common measures of association between two random variables. In the context of matrices, these measures are extended to capture the linear association between entire matrices, rather than just pairs of individual data points.
### Dot product
The dot product is a mathematical operation that takes two vectors and returns a single number. It is calculated by multiplying the corresponding components of the vectors and adding up the results. The dot product is commutative (the order of the vectors doesn't matter), distributive (it distributes over vector addition), and linear with respect to scalar multiplication. It is used in various fields such as mathematics, physics, engineering, computer science, and has applications in vector algebra, geometry, machine learning, signal processing, computer graphics, and other areas.
### SVD
SVD stands for Singular Value Decomposition. It is a matrix factorization technique used in linear algebra that decomposes a given matrix into three matrices, namely, a left singular matrix, a diagonal matrix with singular values, and a right singular matrix. Mathematically, for a given matrix A, SVD can be expressed as:

$$A = U * S * V^T$$

where:
U is a left singular matrix, containing the left singular vectors of A. The columns of U are orthogonal unit vectors.
S is a diagonal matrix, containing the singular values of A. The singular values are non-negative and represent the square root of the eigenvalues of A^T * A or A * A^T.
V^T is the transpose of the right singular matrix, containing the right singular vectors of A. The columns of V^T are orthogonal unit vectors.
SVD has various applications in areas such as matrix computations, linear regression, image processing, dimensionality reduction, data compression, recommendation systems, and others. It is a powerful tool in linear algebra and plays a fundamental role in many numerical and computational tasks.

### Variance
Variance is a statistical measure that quantifies the spread or dispersion of a set of data points. It represents the degree to which individual data points deviate from the mean of the data set. In other words, variance measures how much the data points are scattered around the mean.
Mathematically, variance is calculated as the average of the squared differences between each data point and the mean of the data set. It is denoted by the symbol "Var" or "σ^2" (sigma squared) for a population, and "s^2" for a sample.
The formula for variance is given as:
$$Var = Σ (xi - μ)^2 / N (for population)$$
$$Var = Σ (xi - x̄)^2 / (N-1) (for sample)$$
where:

xi represents each individual data point
μ or x̄ represents the mean of the data set
N represents the total number of data points in the population or sample
A higher variance indicates a wider spread or greater dispersion of data points around the mean, while a lower variance indicates a narrower spread or lesser dispersion. Variance is commonly used in statistics, probability theory, and data analysis to understand and quantify the variability or variability of data sets, which can have implications for decision making, model building, and inferential statistics.

### Task A)
To compute the 100x100 correlation matrix C, we need to understand how to compute the dot product (correlation) between two images represented as columns in the matrix X.
### Task B)
To identify the two most highly correlated and most uncorrelated images, we need to understand how to interpret the values in the correlation matrix and how to plot the corresponding images.
### Task C) 
To compute the 10x10 correlation matrix C, we need to repeat the process from part (a) using a different subset of images.
### Task D)
To create the matrix Y = XX^T and find the first six eigenvectors with the largest magnitude eigenvalue, we need to understand the mathematical properties of eigenvectors and eigenvalues, and how to compute them using matrix operations.
### Task E)
To SVD the matrix X and find the first six principal component directions, we need to understand the mathematical properties of SVD and how to compute it using matrix operations.
### Task F)
To compare the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and compute the norm of difference of their absolute values, we need to understand how to interpret the results of eigenvector and SVD computations, and how to compute the norm of a vector.
### Task G)
To compute the percentage of variance captured by each of the first 6 SVD modes and plot the first 6 SVD modes, we need to understand how to interpret the results of SVD computations and how to visualize the principal component directions.

## Sec. III. Algorithm Implementation and Development
We started by importing the necessary libraries and loaded the dataset using the following code:
```
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load the data
data = scipy.io.loadmat('yalefaces.mat')
X = data['X']
```
We then proceeded to implement the various components of the project as follows:

(a) To compute the 100x100 correlation matrix C, we used the following code:
```
# Compute the correlation matrix
X100 = X[:, :100]
C = np.dot(X100.T, X100)
```
(b) To identify the two most highly correlated and most uncorrelated images, we first computed the correlation matrix C as in part (a), and then used the following code to identify the corresponding images:
```
# Find the indices of the two most highly correlated images
max_corr_idx = np.unravel_index(np.argmax(C - np.eye(C.shape[0]) * C.max()), C.shape)
img1_idx = max_corr_idx[0]
img2_idx = max_corr_idx[1]

# Find the indices of the two most uncorrelated images
min_corr_idx = np.unravel_index(np.argmin(C + np.eye(C.shape[0]) * C.max()), C.shape)
img3_idx = min_corr_idx[0]
img4_idx = min_corr_idx[1]
```
then plotted the corresponding images using the following code:
```
# Plot the two most highly correlated images with switched x and y axes and rotated by 180 degrees
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(np.rot90(X[:, img1_idx].reshape(32, 32), 2), cmap='gray', origin='lower')
axs[0].set_title(f'Image {img1_idx}')
axs[1].imshow(np.rot90(X[:, img2_idx].reshape(32, 32), 2), cmap='gray', origin='lower')
axs[1].set_title(f'Image {img2_idx}')

# Plot the two most uncorrelated images with switched x and y axes and rotated by 180 degrees
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(np.rot90(X[:, img3_idx].reshape(32, 32), 2), cmap='gray', origin='lower')
axs[0].set_title(f'Image {img3_idx}')
axs[1].imshow(np.rot90(X[:, img4_idx].reshape(32, 32), 2), cmap='gray', origin='lower')
axs[1].set_title(f'Image {img4_idx}')
plt.suptitle('Most Uncorrelated Images')

plt.show()
```
(c) To compute the 10x10 correlation matrix X10, we repeated the process from part (a) using a different subset of images:
```
# Select the specified images
image_indices = [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]  # Note that the indices are 0-based
X10 = X[:, image_indices]
```
(d) To create the matrix Y = XX^T and find the first six eigenvectors with the largest magnitude eigenvalue, we used the following code:
```
# Create the matrix Y = X * X^T
Y = np.dot(X, X.T)
# Find the eigenvalues and eigenvectors of Y
eigenvalues, eigenvectors = np.linalg.eig(Y)
# Sort the eigenvectors based on the magnitude of the eigenvalues
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
# Take the first six eigenvectors
top_eigenvectors = sorted_eigenvectors[:, :6]
```
(e) To SVD the matrix X and find the first six principal component directions, we used the following code:

```
# Perform Singular Value Decomposition (SVD) on matrix X
U, S, VT = np.linalg.svd(X, full_matrices=False)
# Take the first six principal component directions
principal_components = U[:, :6]
```
(f) To compare the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and compute the norm of difference of their absolute values, we used the following code:
```
# Extract the first eigenvector v1 from sorted_eigenvectors
v1 = top_eigenvectors[:, 0]

# Extract the first SVD mode u1 from principal_components
u1 = principal_components[:, 0]

# Compute the norm of the difference of their absolute values
norm_diff = np.linalg.norm(np.abs(v1) - np.abs(u1))
```
(g) To compute the percentage of variance captured by each of the first 6 SVD modes and plot the first 6 SVD modes, we used the following code:
```
# Print the percentage of variance captured by each SVD mode
print("Percentage of variance captured by each SVD mode:")
for i in range(6):
    print("SVD mode {}: {:.2f}%".format(i+1, variance_captured[i]))

# Plot the first 6 SVD modes
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].imshow(principal_components[:, i].reshape(32,32), cmap='gray')
    axs[row, col].set_title("SVD mode {}".format(i+1))
    axs[row, col].axis('off')
plt.suptitle("First 6 SVD Modes")
plt.show()
```

## Sec. IV. Computational Results 

### Task A) 
 computed the correlation matrix for a dataset of images using the code provided. The dataset was denoted as "X", and we selected the first 100 images from this dataset to compute the correlation matrix. The resulting correlation matrix was stored in a variable called "C".

To visualize the correlation matrix, we created a plot using matplotlib, a popular Python data visualization library. The plot was created using the "pcolor" function, which generates a color plot of a 2D array. We used the "jet" colormap to represent the correlation values with different colors shown in figure 1

![part a](https://user-images.githubusercontent.com/129792715/232929765-32c042e7-4cdd-4bcc-875d-fcf320f1234b.png)

```
Figure 1, displayed the correlation matrix as a color map. The x-axis and y-axis represented the image indices, ranging from 0 to 99, corresponding to the 100 selected images. The color intensity in the plot represented the strength of the correlation between the images. Higher intensity colors indicated stronger positive or negative correlations, while lower intensity colors indicated weaker or no correlation
```

### Task B)

We loaded a dataset of facial images from the "yalefaces.mat" file, and denoted it as "X". We then computed the correlation matrix for the first 100 images in the dataset. From the correlation matrix, we identified the indices of the two most highly correlated images and the two most uncorrelated images. We visualized the two most highly correlated images and the two most uncorrelated images using subplots with a 2x1 layout for each set of images. The images were displayed with switched x and y axes and rotated by 180 degrees using the "imshow" function from matplotlib with a grayscale colormap. The resulting plots show the images and their corresponding indices, providing insights into the relationships between the facial images in terms of their correlation and lack of correlation. The results are shown in figure 2 and figure 3

![part b](https://user-images.githubusercontent.com/129792715/232930146-707a793c-4223-48fd-8f37-7b889b0d1856.png)
```
figure 2 shows the two most highly correlated images, where the correlation between the images is highest among all pairs of images in the dataset. The images are displayed side-by-side in a 2x1 layout, with their x and y axes switched and rotated by 180 degrees. The titles of the subplots indicate the index of each image in the dataset.
```

![image](https://user-images.githubusercontent.com/129792715/232930164-bc83285e-83db-48ff-bf12-f94a18d7f5b7.png)

```
figure 3 shows he two most uncorrelated images, where the correlation between the images is lowest among all pairs of images in the dataset. Similar to the first set of visualizations, the images are displayed side-by-side in a 2x1 layout, with their x and y axes switched and rotated by 180 degrees. The titles of the subplots indicate the index of each image in the dataset. The overall purpose of these visualizations is to provide insights into the relationships between the facial images in terms of their correlation and lack of correlation, as computed from the correlation matrix.
```

### Task C)
To compute the 10x10 correlation matrix C between images, we used a subset of 10 images and computed the dot product between them. The resulting correlation matrix was plotted using the pcolor function from matplotlib. The result is shown in figure 4
![part c](https://user-images.githubusercontent.com/129792715/232930699-26673299-61e1-4b85-9cc9-c7300ebbb4b3.png)
```
figure 4 show a block diagonal structure, indicating that the images within each block were highly correlated with each other.
```

### Task D)
To create the matrix Y = XX^T and find the first six eigenvectors with the largest magnitude eigenvalue, we used the numpy library to compute the eigenvalues and eigenvectors of the matrix Y. The resulting eigenvectors were then sorted in descending order of eigenvalue magnitude, and the first six eigenvectors were selected. The resulting eigenvectors showed distinct patterns, with each eigenvector capturing a different aspect of the image data.

```
First six eigenvectors:
 [[ 0.02384327  0.04535378  0.05653196  0.04441826 -0.03378603  0.02207542]
 [ 0.02576146  0.04567536  0.04709124  0.05057969 -0.01791442  0.03378819]
 [ 0.02728448  0.04474528  0.0362807   0.05522219 -0.00462854  0.04487476]
 [ 0.0289902   0.04316163  0.02344727  0.05744188  0.00932899  0.05424285]
 [ 0.03057294  0.04080838  0.00992662  0.05720359  0.02096932  0.06259919]
 [ 0.03229324  0.03805116 -0.00241627  0.05372178  0.02766062  0.0694475 ]]
 ```
### Task E)
To SVD the matrix X and find the first six principal component directions, we used the numpy library to compute the singular value decomposition of the matrix X. The resulting principal component directions were then selected, and the first six were plotted. The resulting plots showed distinct patterns, similar to the eigenvectors from part (d), indicating that the principal component directions captured important aspects of the image data.
```
First six principal component directions:
 [[-0.02384327 -0.04535378 -0.05653196  0.04441826 -0.03378603  0.02207542]
 [-0.02576146 -0.04567536 -0.04709124  0.05057969 -0.01791442  0.03378819]
 [-0.02728448 -0.04474528 -0.0362807   0.05522219 -0.00462854  0.04487476]
 [-0.0289902  -0.04316163 -0.02344727  0.05744188  0.00932899  0.05424285]
 [-0.03057294 -0.04080838 -0.00992662  0.05720359  0.02096932  0.06259919]
 [-0.03229324 -0.03805116  0.00241627  0.05372178  0.02766062  0.0694475 ]]
```

### Task F)
 To compare the first eigenvector v1 from part (d) with the first SVD mode u1 from part (e) and compute the norm of difference of their absolute values, we found that the norm difference was relatively small, indicating that the two methods captured similar information about the image
data.

```
Norm of difference of absolute values:  6.415522462814552e-16
```
### Task G)
To compute the percentage of variance captured by each of the first 6 SVD modes and plot the first 6 SVD modes, we found that the first SVD mode captured the majority of the variance in the image data, with the remaining modes capturing progressively less variance. The output is shown in figure 5

![part 6](https://user-images.githubusercontent.com/129792715/232931090-241bfa76-4996-4b5c-aa2e-0d761b496408.png)

```
figure 5  shows that the first six SVD modes captured a significant portion of the variance in the image data, indicating that these modes could be used to represent the image data in a more compact form.
```

## Section V
Based on the computational results from this project, we can conclude that the image data in the yalefaces.mat file has distinct patterns and correlations, which can be captured by various mathematical techniques. Specifically, we found that computing the correlation matrix between images, creating the matrix Y = XX^T and finding the first six eigenvectors, and performing SVD on the matrix X and finding the first six principal component directions were all effective at capturing important information about the image data. Furthermore, we found that the first few SVD modes captured a significant portion of the variance in the image data, indicating that these modes could be used to represent the image data in a more compact form. Overall, this project provides insight into the underlying structure of the image data and demonstrates the effectiveness of various mathematical techniques for analyzing and processing image data.
