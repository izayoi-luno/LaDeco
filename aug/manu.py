import cv2

class ImageClickHandler:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = None
        self.points = []
        self.last_point = None
        self.keep_looping = True

    def load_image(self):
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError("Image not found or path is incorrect.")

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_point = [x, y]
            self.points.append(self.last_point)
            cv2.circle(self.img, self.last_point, 5, (127, 0, 255), -1)
            cv2.imshow("image", self.img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.keep_looping = False

    def show_image(self):
        cv2.namedWindow("image")
        cv2.imshow("image", self.img)
        cv2.setMouseCallback("image", self.click_event)

    def get_clicked_point(self):
        self.show_image()
        while self.keep_looping:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        return self.points

    def run(self):
        try:
            self.load_image()
            self.show_image()
            print("Press 's' to save points and 'q' to quit.")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    print("Saved points:", self.points)
                elif key == ord('q'):
                    self.keep_looping = False
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    handler = ImageClickHandler("./doc/images/sa_40.jpg")
    handler.load_image()
    clicked_point = handler.get_clicked_point()
    print(f"Clicked point: {clicked_point}")