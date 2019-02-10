import cv2
import re
import numpy as np
import imutils
import pytesseract
from skimage.filters import threshold_local
import datetime

now = datetime.datetime.now()
config = ("-l eng --oem 1 --psm 6")

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def ProcessImage(img):
    # Load an color image in grayscale
    # cv2.imshow("Original", orig_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    orig_img = img
    grey_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    H, W = grey_img.shape[:2]
    size_ratio = 8
    # cv2.imshow("Grey Scale", grey_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    resized_img = cv2.resize(grey_img,(int(W/size_ratio), int(H/size_ratio)), interpolation = cv2.INTER_AREA)
    cv2.imshow("resized", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    processed_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
    processed_img = cv2.Canny(processed_img, 75, 200)
    cv2.imshow("Contoured", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(processed_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    screenCnt = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        area = cv2.contourArea(c)
        if area > 1000:

            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    # cont_img = cv2.drawContours(processed_img, [screenCnt], -1, (0, 255, 0), 2)
    cont_img = cv2.drawContours(resized_img, [screenCnt], -1, (0, 255, 0), 2)

    cv2.imshow("resized with contour", cont_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # apply the four point transform to obtain a top-down
    # view of the original image
    screenCnt = screenCnt*size_ratio
    warped_img = four_point_transform(grey_img, screenCnt.reshape(4, 2))
    # cv2.imshow("warped image", warped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #Warped
    kernel = np.ones((2,2),np.uint8)
    warped_img = ~cv2.erode(~warped_img, kernel, iterations=2)
    kernel = np.ones((2,2),np.uint8)
    morph_img = ~cv2.morphologyEx(~warped_img, cv2.MORPH_CLOSE, kernel, iterations=1)


    #Thresh 1
    T = threshold_local(warped_img, 69, offset = 21, method = "gaussian")
    thresh1 = (warped_img > T).astype("uint8") * 255

    #Thresh2
    ret, thresh2 = cv2.threshold(warped_img,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    thresh2 = ~cv2.erode(~thresh2, kernel, iterations=2)

    #Sum
    sum_img = ~cv2.add(~thresh2,~thresh1)

    #Morph
    kernel = np.ones((2,2),np.uint8)
    morph_img = ~cv2.morphologyEx(~sum_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # thresh2 = ~cv2.dilate(~thresh, kernel, iterations=3)
    # kernel = np.ones((2,2),np.uint8)
    # morph_img = ~cv2.erode(~sum_img, kernel, iterations=2)



    thresh1_display = cv2.resize(thresh1,(int(W/size_ratio), int(H/size_ratio)), interpolation = cv2.INTER_AREA)
    thresh2_display = cv2.resize(thresh2,(int(W/size_ratio), int(H/size_ratio)), interpolation = cv2.INTER_AREA)
    morph_display = cv2.resize(morph_img,(int(W/size_ratio), int(H/size_ratio)), interpolation = cv2.INTER_AREA)
    sum_display = cv2.resize(sum_img,(int(W/size_ratio), int(H/size_ratio)), interpolation = cv2.INTER_AREA)
    # warped_display = cv2.resize(warped_img,(int(W/size_ratio), int(H/size_ratio)), interpolation = cv2.INTER_AREA)

    # show the original and scanned images
    # cv2.imshow("thresh1", thresh1_display)
    # cv2.waitKey(0)
    # cv2.imshow("thresh2", thresh2_display)
    # cv2.waitKey(0)
    # cv2.imshow("sum", sum_display)
    # cv2.waitKey(0)
    # cv2.imshow("morph", morph_display)
    # cv2.waitKey(0)
    # cv2.imshow("warped", warped_display)
    # cv2.waitKey(0)
    # cv2.imshow("Original", warped)
    # cv2.waitKey(0)
    return sum_img

img = cv2.imread('long1.jpg')
result = ProcessImage(img)
text1 = pytesseract.image_to_string(result, config=config)
img = cv2.imread('long2.jpg')
result = ProcessImage(img)
text2 = pytesseract.image_to_string(result, config=config)
text = text1 + "\n" + text2

user_file_name = "Vons_Receipt" + "-" + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M') + ".txt"

class Category:
    def __init__(self, name):
        self.name = name
        self.items = dict()
        self.subTotal = 0
        self.length = -1
        self.index = -1

    def UpdateSubTotal(self):
        self.subTotal = 0
        for item, price in self.items.items():
            self.subTotal = round(self.subTotal + price, 2)

    def Print(self):
        rv_str = self.name + "\n"
        print(self.name)

        for item, price in self.items.items():
            print(item + " for $" + str(price))
            rv_str = rv_str + item + ":$" + str(price) + "\n"
        rv_str = rv_str + "Subtotal = $" + str(self.subTotal) + "\n"
        print("Subtotal = $" + str(self.subTotal))
        return rv_str


    def AddItem(self, line):
        rv = re.search(r'.*@\ +(\$)\ *\d.*', line)
        if rv:
            return -1
        rv = re.search(r'.*' + re.escape("BALANCE") + r'.*', line)
        if rv:
            return -1
        rv = re.search(r'.*' + re.escape("Price") + r'.*', line)
        if rv:
            return -1
        rv = re.search(r'.*' + re.escape("Card") + r'.*', line)
        if rv:
            return -1
        rv = re.search(r'\d+\ *(\.|\,)\ *\d+', line)
        if rv:
            price_str = rv.group()
            item = line.replace(price_str, "")
            price = round(float(price_str.replace(",",".")), 2)
            self.items[item] = price
            self.UpdateSubTotal()
            return 0
        else:
            return -1

    def MatchesName(self, line):
        rv = re.search(r'.*' + re.escape(self.name) + r'.*', line)
        if rv:
            return self.name
        return None

    def SetIndex(self, index):
        self.index = index

    def SetLength(self, length):
        self.length = length



output_file = open(user_file_name, "w+")


line_list = text.split("\n")




categories_list = ["MEAT", "REFRIG/FROZEN", "BAKED GOODS", "PRODUCE", "GROCERY"]

categories_dict = dict()
# categories_dict = {"MEAT":[], "FROZEN/REFRIG":[], "BAKED GOODS":[], "GROCERY":[], "PRODUCE":[]}

for category in categories_list:
    categories_dict[category] = Category(category)

for category_name, category in categories_dict.items():
    for index, line in enumerate(line_list):
        name = category.MatchesName(line)
        if name:
            line_list[index] = category_name
            category.SetIndex(index)

for index, line in enumerate(line_list):
    print(str(index) + ":" + line)

for category_name, category in categories_dict.items():
    if category.index == -1:
        index = int(input("What index is " + category_name + " at?"))
        category.SetIndex(index)
        line_list[index] = category_name

index_list = []
for category_name, category in categories_dict.items():
    if category.index >= 0:
        index_list.append(category.index)

prev_index = len(line_list)
for index in sorted(index_list, reverse=True):
    categories_dict[line_list[index]].SetLength(prev_index - index)
    prev_index = index

for category_name, category in categories_dict.items():
    for index in range(category.index, category.index + category.length):
        category.AddItem(line_list[index])

total = 0
for category_name, category in categories_dict.items():
    total = total + category.subTotal
    temp_str = category.Print()
    output_file.write(temp_str)
output_file.write("Total = $" + str(round(total, 2)))
