import cv2
import numpy as np

class ImageClickHandler:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = None
        self.display_img = None
        self.points = []
        self.last_point = None
        self.keep_looping = True
        self.scale_factor = 1.0

    def load_image(self):
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError("Image not found or path is incorrect.")

    def resize_image(self, max_width=800, max_height=600):
        """ Resize image for display while maintaining aspect ratio. """
        height, width = self.img.shape[:2]
        self.scale_factor = min(max_width / width, max_height / height)
        self.display_img = cv2.resize(self.img, None, fx=self.scale_factor, fy=self.scale_factor)

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert the clicked point to the original image scale
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            self.last_point = [orig_x, orig_y, 1]
            self.points.append(self.last_point)
            # Draw on the display image at the clicked point
            cv2.circle(self.display_img, (x, y), 5, (127, 0, 255), -1)
            cv2.imshow("image", self.display_img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.keep_looping = False

    def show_image(self):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        self.resize_image()  # Resize the image for display
        cv2.imshow("image", self.display_img)
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


class BinaryMaskDrawer:
    def __init__(self, img_path, mask_color=(255, 255, 255), brush_size=5):
        """
        Initialize the BinaryMaskDrawer.

        Parameters:
        - img_path: Path to the input image.
        - mask_color: Color of the binary mask (default: white).
        - brush_size: Brush size for drawing on the mask.
        """
        self.image_path = img_path
        self.img = None
        self.display_img = None
        self.mask = None
        self.display_mask = None
        self.mask_color = mask_color
        self.brush_size = brush_size
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.scale_factor = 1.0
        self.keep_looping = True

    def load_image(self):
        """ Load image and create a mask with the same dimensions. """
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError("Image not found or path is incorrect.")
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)  # Initialize mask

    def resize_image(self, max_width=800, max_height=600):
        """ Resize image for display while maintaining aspect ratio. """
        height, width = self.img.shape[:2]
        self.scale_factor = min(max_width / width, max_height / height)
        self.display_img = cv2.resize(self.img, None, fx=self.scale_factor, fy=self.scale_factor)
        self.display_mask = cv2.resize(self.mask, None, fx=self.scale_factor, fy=self.scale_factor)

    def draw_circle(self, event, x, y, flags, param):
        """
        Mouse callback to draw circles on the mask.
        - event: Mouse event.
        - x, y: Coordinates of the event.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Convert the display coordinates to original image coordinates
                orig_x = int(x / self.scale_factor)
                orig_y = int(y / self.scale_factor)
                # Draw circle on original mask
                cv2.circle(self.mask, (orig_x, orig_y), self.brush_size, 255, -1)
                # Draw circle on display image for visual feedback
                cv2.circle(self.display_img, (x, y), self.brush_size, self.mask_color, -1)
                cv2.imshow("image", self.display_img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def show_image(self):
        """ Show image with mouse callback to draw the mask. """
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        self.resize_image()  # Resize the image for display
        cv2.imshow("image", self.display_img)
        cv2.setMouseCallback("image", self.draw_circle)

    def run(self):
        """ Main loop to display image and handle key events. """
        self.load_image()
        self.show_image()

        while self.keep_looping:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to quit
                break
            elif key == ord('s'):  # Save the mask
                print("Mask saved.")
                self.save_mask('binary_mask.png')

        cv2.destroyAllWindows()

    def get_mask(self):
        """ Return the binary mask. """
        return self.mask

    def save_mask(self, out_path):
        """ Save the binary mask to a file. """
        cv2.imwrite(out_path, self.mask)


if __name__ == "__main__":
    # handler = ImageClickHandler("./doc/images/sa_40.jpg")
    # handler.load_image()
    # clicked_point = handler.get_clicked_point()
    # print(f"Clicked point: {clicked_point}")

    mask_drawer = BinaryMaskDrawer("./doc/images/sa_40.jpg", brush_size=40)
    mask_drawer.run()

    binary_mask = mask_drawer.get_mask()
    mask_drawer.save_mask("./sa_40_mask.jpg")