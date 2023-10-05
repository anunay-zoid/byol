import pdqhash
import cv2
import os


image = cv2.imread("/home/anunay/Documents/crop_pad/243.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image2 = cv2.imread("/home/anunay/Documents/crop_pad/225.jpg")
image2= cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

hash_vector1, quality_1 = pdqhash.compute(image)
hash_vector2, quality_2 = pdqhash.compute(image2)

print("image1 hash_vector = == == " , hash_vector1)
print("image_quality_1 ======" , quality_1)

print("image_2 hash_vector = == == " , hash_vector2)
print("image_quality_2 ======" , quality_2)

count=0
for i in range(len(hash_vector2)):
    if hash_vector2[i] == hash_vector1[i]:
        count+=1

print("accuracy ==== ==== ==== " , count/len(hash_vector1))

# Get all the rotations and flips in one pass.
# hash_vectors is a list of vectors in the following order
# - Original
# - Rotated 90 degrees
# - Rotated 180 degrees
# - Rotated 270 degrees
# - Flipped vertically
# - Flipped horizontally
# - Rotated 90 degrees and flipped vertically
# - Rotated 90 degrees and flipped horizontally
# hash_vectors, quality1 = pdqhash.compute_dihedral(image)

# print("all hash-vectors ======",hash_vectors)
# print("quality 1 =======" , quality1)

# # Get the floating point values of the hash.
# hash_vector_float, quality2 = pdqhash.compute_float(image)

# print(hash_vector_float)
# print("quality 2 ======== " , quality2)