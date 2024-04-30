from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os.path as osp
import os, pdb

def create_imgs_grid(images, n_cols, captions, save_path):
    # Unpack grid size
    # rows, cols = grid_size
    n_imgs = len(images)
    rows = round(np.ceil(n_imgs / n_cols).item())
    print(rows, n_cols)
    # Load images
    # images = [Image.open(path) for path in image_paths]
    images = [Image.fromarray(images[i]) for i in range(n_imgs)]
    if captions is None:
        captions = [str(i) for i in range(n_imgs)]
    
    # Assume all images are the same size
    img_width, img_height = images[0].size

    img_width += 10 # add gap between images
    # img_height += 10
    
    # Define the height of the caption area
    caption_height = 50  # You can adjust this value
    
    # Create a new image with extra space for captions
    grid_img = Image.new('RGB', (n_cols * img_width, rows * (img_height + caption_height)), 'white')
    
    # Prepare to draw text
    draw = ImageDraw.Draw(grid_img)
    # use PIL's default)
    font = ImageFont.load_default(size=20)
    
    # Paste images and add captions
    for index, img in enumerate(images):
        row = index // n_cols
        col = index % n_cols
        box = (col * img_width, row * (img_height + caption_height))
        grid_img.paste(img, box)
        text_position = (box[0], box[1] + img_height + 10)  # Adjust vertical position of text
        draw.text(text_position, captions[index], fill="black", font=font)
        img.close()
    
    if save_path is not None:
        # pdb.set_trace()
        os.makedirs(osp.dirname(save_path),exist_ok=True)
        # Save the new image
        grid_img.save(save_path)
        # grid_img.show()
        print(f'[save grid to] {save_path}')
    return np.array(grid_img)