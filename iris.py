import cv2

import numpy as np
import utils

from utils.math_utils import compute_circle_center_coords, euclidean_distance_coords, is_between_parabolas_coords
from utils.error_utils import SUCCESS, WRONG_IMAGE_FORMAT, PUPIL_DETECTION_FAILED, IRIS_DETECTION_FAILED, EYELIDS_DETECTION_FAILED


from math import pi, sin, cos, atan, sqrt, exp, pow
from utils.image_utils import valid_pixel
from utils.error_utils import SUCCESS, SUCCESS, RESOLUTION_ERROR
#--------------------------------------------------------------------------------

#this means that there's a scale difference between the filter size of the median filter
#implemented in the project iris and the one in OpenCV
#ex: project_iris_median(3) = opencv_median(9)      9/3 = 3
#    project_iris_median(9) = opencv_median(27)     27/9 = 3
KERNEL_SIZE_SCALE = 3

#--------------------------------------------------------------------------------

FIRST_MAX_UPPER_BOUND = 170
FIRST_MAX_LOWER_BOUND = 131
FIRST_MAX_DEFAULT_VALUE = 160
FIRST_MAX_OFFSET = 13

#--------------------------------------------------------------------------------

BLACK = 0
WHITE = 255

#--------------------------------------------------------------------------------

PUPIL_THRESHOLD = 0
PUPIL_CIRCLE_THICKNESS = 2
PUPIL_MIN_RADIUS = 10
PUPIL_MAX_RADIUS = 60
PUPIL_RADIUS_INC = 1
PUPIL_CENTER_OFFSET = 5

#--------------------------------------------------------------------------------

IRIS_RADIUS_INC = -5

#--------------------------------------------------------------------------------

BLACK_THRESHOLD = 80
#--------------------------------------------------------------------------------

EYELIDS_POINTS_DISTANCE = 50     # this is the distance between the eyelids points
EYELIDS_PADDING = 21             # i have no idea what it means
EYELID_POINTS_COUNT = 3          # amount of points of an eyelid
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

ANGULAR_RESOLUTION = 256    # default angular resolution
RADIAL_RESOLUTION = 64      # default radial resolution

THRESHOLD_SPECULAR = 220
THRESHOLD_BLACK_BIT = 40    # this is usually the pupil threshold

#--------------------------------------------------------------------------------
TEMPLATE_WIDTH = 256
TEMPLATE_HEIGHT = 64

#--------------------------------------------------------------------------------

GAUSSIAN_SCALE = 0.4770322291

SIN = 0
COS = 1

ENCODED_PIXELS = 1024                   # should be divisible by 32(int) or 64(long)

BITCODE_LENGTH = 2 * ENCODED_PIXELS     # each encoded pixel brings 2 bits to the bitcode

#--------------------------------------------------------------------------------


# hamming distance with masks
def hamming_distance(bit_code_x, mask_x, bit_code_y, mask_y):
    # getting code size (assuming all have the same size)
    code_size = len(bit_code_x)

    # performing logical xor between the two iris codes
    xor_result = np.bitwise_xor(bit_code_x, bit_code_y)

    # performing logical and between the two masks
    and_result = np.bitwise_and(mask_x, mask_y)

    # performing logical and between the two previous results
    result = np.bitwise_and(xor_result, and_result)

    # counting disagreeing bits and normalizing
    return np.count_nonzero(result) / float(code_size)


def encode_iris(norm_img, mask_img, angular_resolution, radial_resolution):
    # getting image dimensions
    height, width = norm_img.shape

    #creating the bitcode and it's mask
    bit_code = np.zeros(BITCODE_LENGTH, np.uint8)
    bit_code_mask = np.zeros(BITCODE_LENGTH, np.uint8)

    # number of slices image is cut up into. Ideally angular slices should divide
    # 360, and size of bitCode without a remainder. More importantly, their product
    # should be divisible by 32
    angular_slices = angular_resolution
    radial_slices = ENCODED_PIXELS // angular_resolution

    # maximum filter size - set to 1/3 of image height to avoid large, uninformative
    # filters
    max_filter = height // 3

    # tracks the position which needs to be modified in the bitcode and bitcodemask
    bit_code_index = 0

    for r_slice in range(radial_slices):
        # works out which pixel in the image to apply the filter to
        # uniformly positions the centres of the filters between radius=3 and radius=height/2
        # does not consider putting a filter centre at less than radius=3, to avoid tiny filters
        radius = ((r_slice * (height - 6)) // (2 * radial_slices)) + 3

        # iet filter dimension to the largest filter that fits in the image
        filter_height = 2 * radius - 1 if radius < (height - radius) else 2 * (height - radius) - 1

        # if the filter size exceeds the width of the image then correct this
        if filter_height > width - 1:
            filter_height = width - 1

        # if the filter size exceeds the maximum size specified earlier then correct this
        if filter_height > max_filter:
            filter_height = max_filter

        # generating sinusoidal filters
        p_sine = generate_sinusoidal_filter(filter_height, SIN)
        p_cosine = generate_sinusoidal_filter(filter_height, COS)

        for a_slice in range(angular_slices):
            theta = a_slice

            bit_code[bit_code_index] = gabor_pixel(radius, theta, p_cosine, norm_img, mask_img)
            bit_code[bit_code_index + 1] = gabor_pixel(radius, theta, p_sine, norm_img, mask_img)

            # check whether the pixel itself is good
            if mask_img[radius, theta]:
                bit_code_mask[bit_code_index] = 1
            else:
                bit_code_mask[bit_code_index] = 0

            # check whether a filter is good or bad
            if not is_good_filter(radius, theta, filter_height, mask_img):
                bit_code_mask[bit_code_index] = 0

            # we're assuming that pairs of bits in the bitCodeMask are equal
            bit_code_mask[bit_code_index + 1] = bit_code_mask[bit_code_index]

            # incrementing the index
            bit_code_index += 2

    return bit_code, bit_code_mask


def generate_sinusoidal_filter(size, sinusoidal_type):
    sum_row = 0.0
    sin_filter = np.empty((size, size), np.float64)

    if sinusoidal_type == SIN:
        wave_fun = lambda phi: sin(pi * phi / (size // 2))

    elif sinusoidal_type == COS:
        wave_fun = lambda phi: cos(pi * phi / (size // 2))

    # unknown filter type
    else:
        return None

    # filling first row
    for j in range(size):
        phi = j - (size // 2)
        wave_value = wave_fun(phi)
        sin_filter.itemset(0, j, wave_value)
        sum_row += wave_value

    # normalizing first row
    for j in range(size):
        old_value = sin_filter.item(0, j)
        sin_filter.itemset(0, j, old_value - (sum_row / size))

    #filling filter
    for i in range(1, size):
        for j in range(size):
            sin_filter.itemset(i, j, sin_filter.item(0, j))

    #generating gaussian filter
    gaussian_filter = generate_gaussian_filter(size)

    #multiplying both filters
    for i in range(size):
        for j in range(size):
            new_value = sin_filter.item(i, j) * gaussian_filter.item(i, j)
            sin_filter.itemset(i, j, new_value)

    # make every row have equal +ve and -ve
    for i in range(size):
        #computing row_sum
        row_sum = 0.0
        for j in range(size):
            row_sum += sin_filter.item(i, j)

        #normalizing
        for j in range(size):
            old_value = sin_filter.item(i, j)
            sin_filter.itemset(i, j, old_value - (row_sum / size))

    return sin_filter



def generate_gaussian_filter(size, peak=15.0):
    # Scale the constants so that gaussian is always in the same range
    # Uses alpha = dimension * (4sqrt(-ln(1/3)))**-1
    # The gaussian will have the value peak/3 at each of its edges
    # and peak/9 at its corners
    alpha = (size - 1) * GAUSSIAN_SCALE
    beta = alpha
    gaussian_filter = np.empty((size, size), np.float64)

    for i in range(size):
        rho = i - (size / 2)
        for j in range(size):
            phi = j - (size / 2)
            wave_value = peak * exp(-pow(rho, 2.0) / pow(alpha, 2.0)) * exp(-pow(phi, 2.0) / pow(beta, 2.0))
            gaussian_filter.itemset(i, j, wave_value)

    return gaussian_filter
	
	
def gabor_pixel(rho, phi, sinusoidal_filter, norm_img, mask_img):
    # size of the filter to be applied
    filter_size = sinusoidal_filter.shape[0]   # we assume that the filter is sqared

    # running total used for integration
    running_total = 0.0

    # translated co-ords within image (image_x, image_y)
    angles = norm_img.shape[1]

    for i in range(filter_size):
        for j in range(filter_size):
            # actual angular position within the image
            image_y = j + phi - (filter_size // 2)

            # allow filters to loop around the image in the angular direction
            image_y %= angles
            if image_y < 0:
                image_y += angles

            # actual radial position within the image
            image_x = i + rho - (filter_size // 2)

            # if the bit is good then apply the filter and add this to the sum
            if mask_img.item(image_x, image_y):
                running_total += sinusoidal_filter.item(i, j) * norm_img.item(image_x, image_y)

    # return true if +ve and false if -ve
    return 1 if running_total >= 0.0 else 0


def is_good_filter(radius, theta, filter_height, mask):
    good_ratio = 0.5  # ratio of good bits in a good filter

    height, width = mask.shape
    r_lb = max(0, radius - (filter_height // 2))
    r_ub = min(height, radius + (filter_height // 2) + 1)

    t_lb = max(0, theta - (filter_height // 2))
    t_ub = min(width, theta + (filter_height // 2) + 1)

    # check the mask of all pixels within the range of the filter
    ratio = np.average(mask[r_lb:r_ub, t_lb:t_ub])

    # if the ratio of good pixels to total pixels in the filter is good, return true
    return ratio >= good_ratio


def normalize_iris(img, angular_resolution, radial_resolution, pupil_center, pupil_radius, iris_center, iris_radius, upper_eyelid, lower_eyelid):
    angles = angular_resolution          # amount of angles to map (<= 360)
    radii = radial_resolution            # iris width (the width of the iris ring)
    radii2 = radii + 2                   # radial resolution plus 2

    img_height = img.shape[0]   # getting the image height
    xp, yp = pupil_center       # (xp, yp) is the pupil center
    xi, yi = iris_center        # (xi, yi) is the iris center
    rp = pupil_radius           # rp is the radius of the pupil
    ri = iris_radius            # ri is the radius of the iris

    # creating the normalized image
    norm_image = np.empty((radii, angles), np.uint8)
    mask_image = np.empty((radii, angles), np.uint8)     # 1 if valid pixel, 0 otherwise

    # computing centers offset
    ox = xp - xi    # offstet of pupil and iris centers in the x axis
    oy = yp - yi    # offstet of pupil and iris centers in the y axis

    if ox < 0:
        sgn = -1
        phi = atan(oy / ox)
    elif ox > 0:
        sgn = 1
        phi = atan(oy / ox)
    else:
        sgn = 1 if oy > 0 else -1
        phi = pi / 2.0

    #computing alpha
    alpha = ox * ox + oy * oy

    #ToDo: Python code here. Optimize it with cython or anything like that.
    #foreach angle
    for col in range(angles):
        # computing the current angle
        theta = col * (2 * pi) / (angles - 1)       # simple "three rule" (for the cubans)
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        # computing beta
        beta = sgn * cos(pi - phi - theta)

        # computing the radius of the iris ring for angle theta (see Libor Masek's thesis)
        r_prime = sqrt(alpha) * beta + sqrt(alpha * beta * beta - (alpha - ri * ri))
        r_prime -= rp

        #foreach radius
        for row in range(radii2):
            # computing radius from pupil center to the current sampled point
            r = rp + r_prime * row / (radii2 - 1)

            #excluding the first and last rows (pupil/iris border and iris/sclera border)
            if 0 < row < radii2 - 1:
                #getting pixel location in the original image
                x = int(xp + r * cos_theta)
                y = int(yp - r * sin_theta)

                #getting the pixel value
                pixel_value = img.item(y, x)    # indexed first by rows, then by columns

                # If pixel out of bounds
                if not valid_pixel(img, x, y):
                    mask_image.itemset(row - 1, col, 0)

                # If pixel is a black bit
                elif pixel_value < THRESHOLD_BLACK_BIT:
                    mask_image.itemset(row - 1, col, 0)

                # If pixel is a specular reflection
                elif pixel_value > THRESHOLD_SPECULAR:
                    mask_image.itemset(row - 1, col, 0)

                # If pixel doesn't belong inside the two parabolas
               # elif not is_between_parabolas_coords(upper_eyelid, lower_eyelid, x, img_height - y):
               #     mask_image.itemset(row - 1, col, 0)

                # Everything is OK
                else:
                    mask_image.itemset(row - 1, col, 1)

                #setting the pixel in the normalized iris image
                norm_image.itemset(row - 1, col, pixel_value)

    #returning the unwrapped image and the corresponding mask
    return norm_image , mask_image


def find_pupil(img):
    #creating a copy of the original image
    copy = img.copy()

    #applying median filter
    cv2.medianBlur(copy, 9 * KERNEL_SIZE_SCALE, copy)

    #computing pupil threshold
    PUPIL_THRESHOLD = get_pupil_threshold(img)

    #applying binary threshold
    cv2.threshold(copy, PUPIL_THRESHOLD, WHITE, cv2.THRESH_BINARY, copy)

    #ToDo: Python code here. Optimize it with cython or anything like that.
    sumx = 0
    sumy = 0
    amount = 0
    height, width = copy.shape
    for x in range(width):
        for y in range(height):
            #black pixel
            if not copy.item(y, x):
                sumx += x
                sumy += y
                amount += 1

    # If sumx and sumy are 0, that means that the filter destroyed the pupil, so
    # autodetection failed
    if sumx == 0 or sumy == 0:
        return None

    sumx //= amount
    sumy //= amount

    radius = 0
    i = sumy
    j = sumx
    # starting from the center and going right
    while not copy.item(i, j):
        radius += 1
        j += 1

    # 2 of padding
    radius -= 2

    #ToDo: Read below.
    # Here i'm going to draw a circle instead of applying sobel again. Instead of
    # detecting the circle in the image i think i should resize the image to a
    # smaller size so the process of finding a circle is less computationally
    # expensive

    white_img = np.ones(copy.shape, np.uint8) * 255
    cv2.circle(white_img, (sumx, sumy), radius, BLACK, PUPIL_CIRCLE_THICKNESS)

    #ToDo: Optimize here using Circle Hough Transform from OpenCV and not this function
    center_rect = (sumx - 1, sumx + 1, sumy - 1, sumy + 1)
    pupil_data = find_circle(white_img, center_rect, radius, radius + 4)

    #if find_circle had some troubles
    if pupil_data is None:
        return None

    p1, p2, p3 = pupil_data
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    xc, yc = compute_circle_center_coords(x1, y1, x2, y2, x3, y3)
    pupil_radius = euclidean_distance_coords(xc, yc, x1, y1)  # could have been p2 or p3

    return (xc, yc), int(pupil_radius + PUPIL_RADIUS_INC)


#the img is the binary thing, not the original
def find_iris(img, pupil_center):
    xc, yc = pupil_center
    center_rect = (xc - PUPIL_CENTER_OFFSET, xc + PUPIL_CENTER_OFFSET,
                   yc - PUPIL_CENTER_OFFSET, yc + PUPIL_CENTER_OFFSET)
    height, width = img.shape

    iris_data = find_circle(img, center_rect, width // 4, height // 2)

    #if find_circle had some troubles
    if iris_data is None:
        return None

    p1, p2, p3 = iris_data
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    xc, yc = compute_circle_center_coords(x1, y1, x2, y2, x3, y3)
    iris_radius = euclidean_distance_coords(xc, yc, x1, y1)  # could have been p2 or p3

    return (xc, yc), int(iris_radius + IRIS_RADIUS_INC)


#This is practically the same implementation of project iris

def find_eyelids(img, pupil_center):
    #performing the "otsu" thresholding
    eyelids = img.copy()

    #performing median blur
    cv2.medianBlur(src=eyelids, ksize=9 * KERNEL_SIZE_SCALE, dst=eyelids)

    #performing threshold with the "no black" index
    no_black_threshold = get_threshold_without_black(eyelids)
    cv2.threshold(src=eyelids, thresh=no_black_threshold, maxval=WHITE, type=cv2.THRESH_BINARY, dst=eyelids)

    #performing median blur again
    eyelids = cv2.medianBlur(src=eyelids, ksize=9 * KERNEL_SIZE_SCALE, dst=eyelids)

    #finding eyelids
    xc, _ = pupil_center               # pupil_center is a tuple with 2 elements
    width = xc
    height = eyelids.shape[0] // 2 - 1   # shape[0] is the amount of rows of the array
    point_dist = EYELIDS_POINTS_DISTANCE
    padding = EYELIDS_PADDING

    upper_eyelid = [None, None, None]
    lower_eyelid = [None, None, None]

    #ToDo: Python code here. Optimize it with cython or anything like that.
    for i in range(height):

        # First eyelid (top-down)
        for j in range(EYELID_POINTS_COUNT):
            dist_along = width - point_dist + (j * point_dist)
            if not upper_eyelid[j] and eyelids.item(i, dist_along + 50) != WHITE:
                upper_eyelid[j] = (dist_along, i + padding)

        # second eyelid (bottom-up)
        for j in range(EYELID_POINTS_COUNT):
            dist_along = width - point_dist + (j * point_dist)
            if not lower_eyelid[j] and eyelids.item(2 * height - i, dist_along) != WHITE:
                lower_eyelid[j] = (dist_along, 2 * height - i)

    #cheking if eyelids were found
    for i in range(EYELID_POINTS_COUNT):
        if not upper_eyelid[i] or not lower_eyelid[i]:
            return None

    #assuming it always went well
    return tuple(upper_eyelid), tuple(lower_eyelid)


#ToDo: This is a Circle Hough Transform implementation. Use the one in OpenCV
#ToDo: Python code here. Optimize it with cython or anything like that.

def get_pupil_threshold(img):
    hist = build_histogram(img)
    pupil_max = -1
    pupil_max_index = -1

    for i in range(90):
        current_value = hist.item(i)

        if current_value > pupil_max:
            pupil_max = current_value
            pupil_max_index = i

    return pupil_max_index + 8


#ToDo: Python code here. Optimize it with cython or anything like that.

def build_histogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])
	
def get_threshold_without_black(img):
    #computing histogram
    hist = build_histogram(img)

    #computing number of pixels with value >= 80
    pixel_count = 0
    for i in range(BLACK_THRESHOLD, 256):
        pixel_count += hist.item(i)

    #computing the probability distribution from histogram
    p = compute_probability_distribution(hist, pixel_count)

    #returning the max goodness index
    return get_max_goodness_index(p)


#ToDo: Python code here. Optimize it by using cython or anything like that.
#ToDo: Do all in a single pass-omega, mew, etc- (i want this code to resemble the project iris source code)

def compute_probability_distribution(hist, total_pixels):
    prob_dist = np.empty(256, dtype=np.float32)
    total = float(total_pixels)

    for i in range(256):
        if i < BLACK_THRESHOLD:
            prob_dist.itemset(i, 0.0)
        else:
            prob_dist.itemset(i, hist.item(i) / total)

    return prob_dist


#ToDo: Python code here. Optimize it by using cython or anything like that.

def get_max_goodness_index(prob_dist):
    #compute goodness array
    goodness = np.empty(256, np.float32)
    mew_256 = compute_mew(256, prob_dist)
    for k in range(256):
        wk = compute_omega(k, prob_dist)
        mk = compute_mew(k, prob_dist)

        if wk == 0 or wk == 1:
            goodness_value = 0
        else:
            factor = mew_256 * wk - mk
            num = factor * factor
            den = wk * (1.0 - wk)
            goodness_value = num / den

        goodness.itemset(k, goodness_value)

    #get max index (this can be done in the previous step)
    max_index = 0
    for i in range(256):
        if goodness.item(i) > goodness.item(max_index):
            max_index = i

    return max_index


#ToDo: Python code here. Optimize it by using cython or anything like that.

def compute_mew(k, prob_dist):
    result = 0.0

    for i in range(k):
        result += (i + 1) * prob_dist.item(i)

    return result
def compute_omega(k, prob_dist):
    result = 0.0

    for i in range(k):
        result += prob_dist.item(i)

    return result


#ToDo: Python code here. Optimize it by using cython or anything like that.
	
def find_circle(img, centerRegion, minRadius, maxRadius):
    #centerRegion is the rectangle for iris center

    if centerRegion is None:
        height, width = img.shape
        # (x_left, x_right, y_top, y_bottom) ->the whole image
        centerRegion = (0, width - 1, 0, height - 1)

    x_left, x_right, y_top, y_bottom = centerRegion
    a_min = x_left
    a_max = x_right
    b_min = y_top
    b_max = y_bottom

    r_min = minRadius
    r_max = maxRadius

    a = a_max - a_min
    b = b_max - b_min
    r = r_max - r_min

    # The sum of the values which are >= maxVotes - 1
    total_r = 0
    total_a = 0
    total_b = 0

    # Amount of values that are >= maxVotes - 1
    amount = 0

    # The max amount of votes that any has
    maxVotes = 0

    # Create and initialise accumulator to 0
    acc = np.zeros((a, b, r), np.int32)

    # For each black point, find the circles which satisfy the equation where the
    # parameters are limited by a,b and r.
    height, width = img.shape
    for x in range(width):
        for y in range(height):
            #if pixel is white continue
            if img.item(y, x):
                continue

            for _a in range(a):
                for _b in range(b):
                    for _r in range(r):
                        sq_a = x - (_a + a_min)
                        sq_b = y - (_b + b_min)
                        sq_r = r - (_r + r_min)

                        if sq_a * sq_a + sq_b * sq_b == sq_r * sq_r:
                            new_value = acc.item((_a, _b, _r)) + 1
                            acc.itemset((_a, _b, _r), new_value)

                            if new_value >= maxVotes:
                                maxVotes = new_value

    for _a in range(a):
        for _b in range(b):
            for _r in range(r):
                if acc.item((_a, _b, _r)) >= maxVotes - 1:
                    total_a += _a + a_min
                    total_b += _b + b_min
                    total_r += _r + r_min
                    amount += 1

    # Get the initial average values
    top_a = total_a / amount
    top_b = total_b / amount
    top_r = total_r / amount

    # Returning the three points
    p1 = (top_a + top_r, top_b)
    p2 = (top_a - top_r, top_b)
    p3 = (top_a, top_b + top_r)

    return p1, p2, p3


#ToDo: Python code here. Optimize it with cython or anything like that.

# hamming distance with masks
def hamming_distance(bit_code_x, mask_x, bit_code_y, mask_y):
    # getting code size (assuming all have the same size)
    code_size = len(bit_code_x)

    # performing logical xor between the two iris codes
    xor_result = np.bitwise_xor(bit_code_x, bit_code_y)

    # performing logical and between the two masks
    and_result = np.bitwise_and(mask_x, mask_y)

    # performing logical and between the two previous results
    result = np.bitwise_and(xor_result, and_result)

    # counting disagreeing bits and normalizing
    return np.count_nonzero(result) / float(code_size)

	
	
eye_img = cv2.imread('C:\\Users\\patel\\Documents\\Iris-Recognition-master\\CASIA1\\2\\002_1_2.jpg')
eye_img1 = cv2.imread('C:\\Users\\patel\\Documents\\Iris-Recognition-master\\CASIA1\\3\\003_1_3.jpg')
def gt(eye_img):

	gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

	# ----------------------------------------------------------------------------
	#trying to detect pupil (from original image)
	pupil_data = find_pupil(gray)
				
	# ----------------------------------------------------------------------------
	#detecting iris
	iris_data = find_iris(gray, pupil_data[0])  # pupil_data[0] = pupil center
	# ----------------------------------------------------------------------------
	#detecting eyelids
	eyelids_data = find_eyelids(gray, pupil_data[0])
	# ----------------------------------------------------------------------------

	#setting pupil control points
	pupil_center = pupil_data[0]
	pupil_radius = pupil_data[1]
	iris_center = iris_data[0]
	iris_radius = iris_data[1]
	upper_eyelid, lower_eyelid = eyelids_data
	width = TEMPLATE_WIDTH
	height = TEMPLATE_HEIGHT
	normal,mask = normalize_iris(gray, width, height, pupil_center, pupil_radius, iris_center, iris_radius, upper_eyelid, lower_eyelid)
	radres, angres = normal.shape
	nomal_bit,mask_bit = encode_iris(normal, mask, angres, radres)
	return nomal_bit, mask_bit
	
def recogn(eye_img):
	for i in range(1,10):
		for j in range(1, 3):
			for k in range(1, 5):
				if j == 1 and k==4:
					continue
				eye_img1 = cv2.imread('C:\\Users\\patel\\Documents\\Iris-Recognition-master\\CASIA1\\'+str(i)+'\\00'+str(i)+'_'+str(j)+'_'+str(k)+'.jpg')
				x,y= gt(eye_img)
				x1,y1=gt(eye_img1)
				result= hamming_distance(x,y,x1,y1)
				print(i,j,k,result)
				if result==0.0:
					return i
	else:
		return -1
print(recogn(eye_img))


		
