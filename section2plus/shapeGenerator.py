import os
import random
from PIL import Image, ImageDraw

# ================= تنظیمات =================
NUM_IMAGES = 100
NUM_SQUARES = 68       # تعداد عکس‌هایی که قطعاً مربع دارند
IMG_SIZE = 512         # ابعاد تصاویر (512x512)
SQUARE_SIZE = 48       # سایز مربع هدف (دقیقاً برابر با Template کد C)
OUTPUT_DIR = "dataset" # نام پوشه خروجی
# ============================================

# ایجاد پوشه dataset اگر وجود نداشته باشد
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ایجاد یک لیست شامل 68 تا True (دارای مربع) و 32 تا False (بدون مربع)
# و سپس بر هم زدن (Shuffle) آن تا توزیع عکس‌ها کاملاً تصادفی باشد
has_square_list = [True] * NUM_SQUARES + [False] * (NUM_IMAGES - NUM_SQUARES)
random.shuffle(has_square_list)

def get_random_color():
    """تولید یک رنگ تصادفی روشن برای اشیاء"""
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

print(f"Generating {NUM_IMAGES} images in '{OUTPUT_DIR}' folder...")

for i in range(1, NUM_IMAGES + 1):
    # ایجاد یک تصویر با پس‌زمینه تیره (نزدیک به مشکی برای عملکرد بهتر لبه‌یابی سوبل)
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    is_target = has_square_list[i-1]
    
    # -----------------------------------------------------------------
    # رسم اشکال مزاحم (مستطیل‌های غیرمربع و دایره‌ها) در همه عکس‌ها
    # -----------------------------------------------------------------
    num_distractors = random.randint(2, 6)
    for _ in range(num_distractors):
        x = random.randint(10, IMG_SIZE - 100)
        y = random.randint(10, IMG_SIZE - 100)
        shape_type = random.choice(['rectangle', 'ellipse'])
        
        if shape_type == 'rectangle':
            # تولید مستطیل (اطمینان از اینکه طول و عرض برابر نیستند تا مربع نشود)
            w = random.randint(30, 100)
            h = random.randint(30, 100)
            if w == h: 
                w += 20 
            draw.rectangle([x, y, x + w, y + h], fill=get_random_color())
        else:
            # تولید دایره / بیضی
            r1 = random.randint(30, 90)
            r2 = random.randint(30, 90)
            draw.ellipse([x, y, x + r1, y + r2], fill=get_random_color())

    # -----------------------------------------------------------------
    # رسم مربع‌های هدف (فقط در 68 عکسی که انتخاب شده‌اند)
    # -----------------------------------------------------------------
    if is_target:
        # بین 1 تا 3 مربع در عکس قرار می‌دهیم
        num_squares_to_draw = random.randint(1, 3)
        for _ in range(num_squares_to_draw):
            x = random.randint(10, IMG_SIZE - (SQUARE_SIZE + 10))
            y = random.randint(10, IMG_SIZE - (SQUARE_SIZE + 10))
            
            # رسم مربع با سایز دقیق 48x48
            draw.rectangle([x, y, x + SQUARE_SIZE, y + SQUARE_SIZE], fill=get_random_color())

    # ذخیره تصویر با فرمت PPM (فرمت مورد نیاز کد C)
    filename = os.path.join(OUTPUT_DIR, f"image_{i}.ppm")
    img.save(filename)

print("\nDataset generation completed successfully!")
print(f"Total Images: {NUM_IMAGES}")
print(f"Images WITH Squares: {NUM_SQUARES}")
print(f"Images WITHOUT Squares: {NUM_IMAGES - NUM_SQUARES}")
