import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import torchvision.transforms.functional as TF
# import pywt


class ImageTransformer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.output_dir = './Images/'
        # os.makedirs(self.output_dir, exist_ok=True)
        self.original_image = Image.open(self.image_path).convert("RGB")
        self.cv2_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)

    def save_image(self, image, name, use_cv2=False):
        path = os.path.join(self.output_dir, f"{name}.jpg")
        if use_cv2:
            cv2.imwrite(path, image)
        else:
            image.save(path)
        print(f"Saved: {path}")

    def brightness_adjust(self):
        enhancer = ImageEnhance.Brightness(self.original_image)
        bright_img = enhancer.enhance(1.3)
        return Image.fromarray(bright_img)

    def contrast_adjust(self):
        enhancer = ImageEnhance.Contrast(self.original_image)
        contrast_img = enhancer.enhance(1.5)
        return Image.fromarray(contrast_img)

    def color_jitter(self):
        enhancer = ImageEnhance.Color(self.original_image)
        color_img = enhancer.enhance(1.4)
        # self.save_image(color_img, "color_jitter")
        return Image.fromarray(color_img)

    def gaussian_noise(self):
        img = np.array(self.original_image)
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        # self.save_image(Image.fromarray(noisy_img), "gaussian_noise")
        return Image.fromarray(noisy_img)

    def salt_and_pepper_noise(self):
        img = np.array(self.original_image)
        s_vs_p = 0.5
        amount = 0.02
        out = np.copy(img)

        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[tuple(coords)] = 255

        num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[tuple(coords)] = 0

        # self.save_image(Image.fromarray(out), "salt_pepper")
        return Image.fromarray(out)

    def blur(self):
        blurred = self.original_image.filter(ImageFilter.GaussianBlur(radius=2))
        # self.save_image(blurred, "blur")
        return Image.fromarray(blurred)

    def motion_blur(self):
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        output = cv2.filter2D(self.cv2_image, -1, kernel)
        # self.save_image(output, "motion_blur", use_cv2=True)
        return Image.fromarray(output)

    def jpeg_artifact(self):
        temp_path = os.path.join(self.output_dir, "jpeg_compressed.jpg")
        self.original_image.save(temp_path, "JPEG", quality=10)
        img = Image.open(temp_path)
        # self.save_image(img, "jpeg_artifact")
        return Image.fromarray(img)

    def translate(self):
        rows, cols, _ = self.cv2_image.shape
        M = np.float32([[1, 0, 10], [0, 1, 10]]) 
        dst = cv2.warpAffine(self.cv2_image, M, (cols, rows))
        # self.save_image(dst, "translation", use_cv2=True)
        return Image.fromarray(dst)

    def scale(self):
        scale_percent = 80
        width = int(self.cv2_image.shape[1] * scale_percent / 100)
        height = int(self.cv2_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(self.cv2_image, dim, interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        # self.save_image(padded, "scaling", use_cv2=True)
        return Image.fromarray(padded)

    def crop(self):
        width, height = self.original_image.size
        crop_area = (10, 10, width - 10, height - 10)
        cropped = self.original_image.crop(crop_area)
        # self.save_image(cropped, "cropped")
        return Image.fromarray(cropped)

    
    def random_erasing(self):
        img = np.array(self.original_image)
        h, w, _ = img.shape
        er_size = (int(h*0.2), int(w*0.2))
        x = random.randint(0, h - er_size[0])
        y = random.randint(0, w - er_size[1])
        img[x:x+er_size[0], y:y+er_size[1], :] = np.random.randint(0, 256)
        # self.save_image(Image.fromarray(img), "random_erasing")
        return Image.fromarray(img)

    def hist_equalization(self):
        img = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # self.save_image(res, "clahe", use_cv2=True)
        return Image.fromarray(res)

    def style_augmentation(self):
        img = self.original_image
        std = random.uniform(0.05, 0.15)
        img_np = np.array(img).astype(np.float32)/255.0
        noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np + noise, 0, 1)
        img_styled = Image.fromarray((img_np*255).astype(np.uint8))
        # self.save_image(img_styled, "style_aug")
        return Image.fromarray(img_styled)

    def posterize(self):
        img = ImageOps.posterize(self.original_image, bits=4)
        # self.save_image(img, "posterize")
        return Image.fromarray(img)

    def gridmask(self):
        img = np.array(self.original_image)
        h, w, c = img.shape
        d = 32
        mask = np.ones((h, w), dtype=bool)
        for y in range(0, h, d):
            for x in range(0, w, d):
                if np.random.rand() < 0.5:
                    mask[y:y+d//2, x:x+d//2] = False
        img[~mask] = 0
        # self.save_image(Image.fromarray(img), "gridmask")
        return Image.fromarray(img)

    def elastic_transform(self, alpha=34, sigma=4):
        img = np.array(self.original_image)
        random_state = np.random.RandomState(None)
        shape = img.shape[:2]
        dx = (random_state.rand(*shape) * 2 - 1) * alpha
        dy = (random_state.rand(*shape) * 2 - 1) * alpha
        dx = cv2.GaussianBlur(dx, (17,17), sigma)
        dy = cv2.GaussianBlur(dy, (17,17), sigma)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        remap = cv2.remap(img, (x+dx).astype(np.float32), (y+dy).astype(np.float32),
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # self.save_image(Image.fromarray(remap), "elastic_transform")
        return Image.fromarray(remap)

    def glass_blur(self):
        img = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
        sigma = 1
        blurred = cv2.GaussianBlur(img, (0,0), sigma)
        for _ in range(3):
            i = np.random.randint(0, img.shape[0]-2)
            j = np.random.randint(0, img.shape[1]-2)
            img[i:i+3, j:j+3] = blurred[i:i+3, j:j+3]
        result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # self.save_image(result, "glass_blur")
        return Image.fromarray(result)

    def sharpen(self):
        img = self.original_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        # self.save_image(img, "sharpen")
        return Image.fromarray(img)

    def freq_mask(self):
        img = np.array(self.original_image).astype(float)/255
        freq = np.fft.fft2(img, axes=(0,1))
        amp, phase = np.abs(freq), np.exp(1j*np.angle(freq))
        mask = np.ones_like(amp)
        mask[amp < np.percentile(amp,10)] = 0
        img_f = (amp * mask) * phase
        img_r = np.clip(np.abs(np.fft.ifft2(img_f,axes=(0,1)))*255,0,255).astype('uint8')
        # self.save_image(Image.fromarray(img_r), "freq_mask")
        return Image.fromarray(img_r)

    def median_blur(self):
        img = cv2.medianBlur(self.cv2_image, 5)
        # self.save_image(img, "median_blur", use_cv2=True)
        return Image.fromarray(img)

    def avg_blur(self):
        img = cv2.blur(self.cv2_image, (5,5))
        # self.save_image(img, "avg_blur", use_cv2=True)
        return Image.fromarray(img)

    def bilateral_blur(self):
        img = cv2.bilateralFilter(self.cv2_image, d=9, sigmaColor=75, sigmaSpace=75)
        # self.save_image(img, "bilateral_blur", use_cv2=True)
        return Image.fromarray(img)

    def pooling(self, mode="max", size=2):
        h, w, c = self.cv2_image.shape
        img = self.cv2_image.reshape(h//size, size, w//size, size, c)
        if mode=="average":
            pooled = img.mean(axis=(1,3))
        elif mode=="max":
            pooled = img.max(axis=(1,3))
        elif mode=="min":
            pooled = img.min(axis=(1,3))
        pooled = cv2.resize(pooled.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        # self.save_image(pooled, f"{mode}_pool", use_cv2=True)
        return Image.fromarray(pooled)

    def random_deletion(self, prob=0.05):
        img = self.cv2_image.copy()
        mask = np.random.rand(*img.shape[:2]) < prob
        img[mask] = 0
        # self.save_image(img, "random_deletion", use_cv2=True)
        return Image.fromarray(img)

    def random_conv(self, kernel_size=3):
        kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
        kernel /= np.sum(np.abs(kernel)) + 1e-6
        blurred = cv2.filter2D(self.cv2_image, -1, kernel)
        # self.save_image(blurred, "random_conv", use_cv2=True)
        return Image.fromarray(blurred)

    def morphology_patch(self, op='erode', k=3, patch_size=50):
        img = self.cv2_image.copy()
        h, w = img.shape[:2]
        x, y = np.random.randint(0, w-patch_size), np.random.randint(0, h-patch_size)
        patch = img[y:y+patch_size, x:x+patch_size]
        kernel = np.ones((k, k), np.uint8)
        if op == 'erode':   
            patch2 = cv2.erode(patch, kernel, iterations=1)
        else:
            patch2 = cv2.dilate(patch, kernel, iterations=1)
        img[y:y+patch_size, x:x+patch_size] = patch2
        # self.save_image(img, f"{op}_patch", use_cv2=True)
        return Image.fromarray(img)

    def add_shadow(self):
        img = np.array(self.original_image).astype(float)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=float)
        cx, cy = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
        rx, ry = w//3, h//3
        y, x = np.ogrid[:h, :w]
        gauss = np.exp(-(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))
        gauss = 1 - 0.5 * gauss  # darken region
        for c in range(3):
            img[..., c] *= gauss
        img = np.clip(img, 0, 255).astype(np.uint8)
        # self.save_image(Image.fromarray(img), "add_shadow")
        return Image.fromarray(img)


    def low_high_pass_mix(self, alpha=0.7):
        img = self.cv2_image.astype(float)
        blur = cv2.GaussianBlur(img, (15,15), 5)
        high = img - blur
        mixed = np.clip(alpha*img + (1 - alpha)*high, 0, 255).astype(np.uint8)
        # self.save_image(mixed, "low_high_mix", use_cv2=True)
        return Image.fromarray(mixed)

    def add_specular(self):
        img = np.array(self.original_image).astype(float)
        h, w = img.shape[:2]
        cx, cy = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
        sigma = min(h, w) // 20
        y, x = np.ogrid[:h, :w]
        blob = np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))
        blob = blob[:, :, None]
        spec = np.clip(img + 150 * blob, 0, 255).astype(np.uint8)
        # self.save_image(Image.fromarray(spec), "add_specular")
        return Image.fromarray(spec)

    def nl_means(self):
        img = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        l2 = cv2.fastNlMeansDenoising(l, None, 10, 7, 21)
        merged = cv2.merge((l2, a, b))
        denoised = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        # self.save_image(denoised, "nl_means", use_cv2=True)
        return Image.fromarray(denoised)

    def poisson_noise(self):
        img = np.array(self.original_image).astype(np.float32) / 255.0
        noisy = np.random.poisson(img * 255) / 255.0
        noisy = np.clip(noisy, 0, 1)
        result = (noisy * 255).astype(np.uint8)
        # self.save_image(Image.fromarray(result), "poisson_noise")
        return Image.fromarray(result)

    def speckle_noise(self):
        img = np.array(self.original_image).astype(np.float32) / 255.0
        noise = img * np.random.randn(*img.shape) * 0.2
        noisy = np.clip(img + noise, 0, 1)
        result = (noisy * 255).astype(np.uint8)
        # self.save_image(Image.fromarray(result), "speckle_noise")
        return Image.fromarray(result)
        
    def impulse_noise(self, prob=0.01):
        img = self.cv2_image.copy()
        mask = np.random.rand(*img.shape[:2]) < prob
        img[mask] = 255
        # self.save_image(img, "impulse_noise", use_cv2=True)
        return Image.fromarray(img)

    def coarse_dropout(self, size=30, num_blocks=5):
        img = self.cv2_image.copy()
        h, w = img.shape[:2]
        for _ in range(num_blocks):
            x = np.random.randint(0, w-size)
            y = np.random.randint(0, h-size)
            img[y:y+size, x:x+size] = 0
        # self.save_image(img, "coarse_dropout", use_cv2=True)
        return Image.fromarray(img)

    def mean_shift(self):
        img = cv2.pyrMeanShiftFiltering(self.cv2_image, sp=10, sr=20)
        # self.save_image(img, "mean_shift", use_cv2=True)
        return Image.fromarray(img)

    def defocus_blur(self, radius=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        blurred = cv2.filter2D(self.cv2_image, -1, kernel / kernel.sum())
        # self.save_image(blurred, "defocus_blur", use_cv2=True)
        return Image.fromarray(blurred)


if __name__ == "__main__":  
    transformer = ImageTransformer("sample.jpg", output_dir="transformed_outputs")
    