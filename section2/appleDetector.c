#include <immintrin.h> // برای استفاده از دستورات برداری اینتل (AVX/SSE) یا همان Intrinsics
#include <stdio.h>     
#include <stdlib.h>    
#include <time.h>      
#include <float.h>

// ساختار داده‌ای برای نگهداری اطلاعات تصویر
typedef struct {
    int width;           // عرض تصویر
    int height;          // ارتفاع تصویر
    unsigned char *data; // آرایه یک بعدی برای ذخیره پیکسل‌ها (به صورت R G B پشت سر هم)
} Image;
// تابعی برای خواندن تصاویر با فرمت ppm با هدر P6 که در آن مقدار rgb پسکیل ها به صورت بایتی بیان شده
Image *readPicture(const char *filename) {
    FILE *file = fopen(filename, "rb"); // باز کردن فایل به صورت باینری (read binary)
    if (!file) {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }
    char buffer[16];
    fgets(buffer, sizeof(buffer), file); // خواندن هدر فرمت فایل (معملاً "P6")
    
    // رد کردن کامنت‌های احتمالی در فایل عکس (خطوطی که با # شروع می‌شوند)
    int c = getc(file);
    while (c == '#') {
        while (getc(file) != '\n');
        c = getc(file);
    }
    ungetc(c, file); // برگرداندن آخرین کاراکتر خوانده شده (چون کامنت نبوده)
    
    int width, height, maxVal;
    fscanf(file, "%d %d", &width, &height); // خواندن طول و عرض عکس
    fscanf(file, "%d", &maxVal);            // خواندن حداکثر مقدار رنگ (معمولا 255)
    fgetc(file);                            // رد کردن کاراکتر اینتر (Newline)
    
    // تخصیص حافظه برای ساختار تصویر و آرایه پیکسل‌ها
    Image *image = (Image *)malloc(sizeof(Image));
    image->width = width;
    image->height = height;
    // ضرب در 3 به این خاطر است که هر پیکسل 3 بایت دارد (Red, Green, Blue)
    image->data = (unsigned char *)malloc(width * height * 3);
    
    // خواندن کل دیتای باینری عکس به صورت یکجا و ریختن در حافظه
    fread(image->data, 1, width * height * 3, file);
    fclose(file);
    return image;
}

void writePicture(Image *image, const char *filename) {
    FILE *file = fopen(filename, "wb"); // باز کردن فایل برای نوشتن باینری
    // نوشتن هدر استاندارد P6
    fprintf(file, "P6\n%d %d\n255\n", image->width, image->height);
    // نوشتن کل آرایه پیکسل‌ها در فایل
    fwrite(image->data, 1, image->width * image->height * 3, file);
    fclose(file);
}


//تبدیل عکس رنگی به خاکستری
float* convert_to_grayscale_float(Image *img) {
    // چون برای این کانولوشن و تشخیص شی رنگ برام ممهم نیست و همچنین مجاسبه با استفاده از rgb  سخت تر و پیچیده تره همون اول عکس رو grayscale میکنم
    float* gray_img = (float*)malloc(img->width * img->height * sizeof(float));
    
    for (int i = 0; i < img->width * img->height; i++) {
        int idx = i * 3; // پیدا کردن ایندکس شروع هر پیکسل (چون هر پیکسل 3 بایت است)
        float r = (float)img->data[idx];
        float g = (float)img->data[idx + 1];
        float b = (float)img->data[idx + 2];
        // استفاده از فرمول استاندارد Luminosity برای تبدیل رنگ  های rgb به grayscale
        gray_img[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    return gray_img;
}

// ساخت و تغییر سایز کرنل از روی عکس تمپلیت
float* create_kernel_from_image(const char* filename, int* k_width, int* k_height) {
    Image* tpl_img = readPicture(filename); // خواندن عکس الگو (تمپلیت)
    *k_width = tpl_img->width;
    *k_height = tpl_img->height;
    
    int total_pixels = (*k_width) * (*k_height);
    float* kernel = (float*)malloc(total_pixels * sizeof(float));
    
    for (int i = 0; i < total_pixels; i++) {
        int idx = i * 3;
        float r = (float)tpl_img->data[idx];
        float g = (float)tpl_img->data[idx + 1];
        float b = (float)tpl_img->data[idx + 2];
        
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        // کرنل را میسازم :  اگر پیکسل تاریک باشد (لوگو) مقدار 1+ وگرنه پس‌زمینه 1- می‌گیرد
        kernel[i] = (gray < 128.0f) ? 1.0f : -1.0f;
    }
    
    free(tpl_img->data); free(tpl_img);
    return kernel;
}

// تابع تغییر سایز کرنل (برای پیدا کردن الگو در مقیاس‌های مختلف در عکس اصلی)
float* resize_kernel(const float* original_kernel, int orig_w, int orig_h, int new_w, int new_h) {
    float* resized = (float*)malloc(new_w * new_h * sizeof(float));
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            // روش Nearest Neighbor برای پیدا کردن پیکسل متناظر در کرنل اصلی
            //با این روش کرنل را به سایز های بزرگتر و کوچکتر صرفا با استفاده از تناسب  گسترش میدم
            int src_x = (x * orig_w) / new_w;
            int src_y = (y * orig_h) / new_h;
            // جلوگیری از خطای (Out of bounds)
            if (src_x >= orig_w) src_x = orig_w - 1;
            if (src_y >= orig_h) src_y = orig_h - 1;
            resized[y * new_w + x] = original_kernel[src_y * orig_w + src_x];
        }
    }
    return resized;
}

float templateMatch_C(const float *grayImage, int width, int height, float *kernel, int kWidth, int kHeight, int *best_x, int *best_y) {
    float max_score = -FLT_MAX; // مقدار اولیه روی منفی بی‌نهایت فلوت میگذارم 
    //زیرا میخوام بهترین پنجره ای که یه پیکسل ایجاد میکنه رو به عنوام پیکسلی درون این لولگو یافت شده به دست بیارم برای رسم یه حاشه دورش 

    //با استفاده از دو حلقه اول سعی میکنم ماتریس کرنل را روی پیکسل های عکس ورودی حرکت دهم به طوری که از عکس خارج نشود
    for (int y = 0; y <= height - kHeight; y++) {
        for (int x = 0; x <= width - kWidth; x++) {
            float sum = 0.0f; // متغیر برای جمع امتیازات این پنجره
            
            //ضرب نظیر به نظیر پیکسل‌های درون پنجره با کرنل
            for (int m = 0; m < kHeight; m++) {
                for (int n = 0; n < kWidth; n++) {
                    //استخراج پیکسل متناظر از عکس اصلی
                    float gray = grayImage[(y + m) * width + (x + n)];
                    //استخراج وزن کرنل
                    float kVal = kernel[m * kWidth + n];
                    //ضرب و جمع 
                    sum += kVal * gray;
                }
            }
            //میانگین‌گیری امتیاز برای اینکه سایز کرنل روی نتیجه نهایی تاثیر نگذارد
            float score = sum / (float)(kWidth * kHeight);
            
            //اگر امتیاز این پنجره از بهترین امتیاز قبلی بیشتر بود، ذخیره میکنم
            if (score > max_score) {
                max_score = score;
                *best_x = x; 
                *best_y = y;
            }
        }
    }
    return max_score;
}


float templateMatch_AVX2(const float *grayImage, int width, int height, float *kernel, int kWidth, int kHeight, int *best_x, int *best_y) {
    float max_score = -FLT_MAX;//مقدار اولیه روی منفی بی‌نهایت فلوت میگذارم 

    //حلقه حرکت عمودی پنجره (دست نخورده باقی می‌ماند)
    for (int y = 0; y <= height - kHeight; y++) {
        
        //حلقه افقی را 8 تا 8 تا جلو می‌برم 
        //چون هر رجیستر 256 بیتی AVX می‌تواند 8 عدد Float (32 بیتی) را در خود جای دهد
        //همزمان 8 پنجره کشویی مجاور را در یکبار پردازش محاسبه می‌کنم
        for (int x = 0; x <= width - kWidth - 8; x += 8) {
            
            //معادل اسمبلی: vxorps ymm, ymm, ymm (صفر کردن رجیستر 256 بیتی)
            //این رجیستر قرار است جمع امتیازات 8 پنجره مختلف را نگه دارد
            __m256 v_sums = _mm256_setzero_ps();

            //پیمایش روی پیکسل‌های کرنل
            for (int m = 0; m < kHeight; m++) {
                for (int n = 0; n < kWidth; n++) {
                    
                    //معادل اسمبلی: vmovups ymm, [mem]
                    //بارگذاری 8 پیکسل متوالی از عکس اصلی به صورت Unaligned
                    //این 8 پیکسل، در واقع پیکسلِ [m,n] برای 8 پنجره متفاوت هستند
                    __m256 v_gray = _mm256_loadu_ps(&grayImage[(y + m) * width + (x + n)]);
                    
                    //استخراج یک مقدار وزن از کرنل
                    float kVal = kernel[m * kWidth + n];
                    
                    //معادل اسمبلی: vbroadcastss ymm, xmm
                    //کپی کردن (Broadcast) این یک مقدار کرنل در تمام 8 خانه رجیستر برداری
                    __m256 v_kernel = _mm256_set1_ps(kVal);
                    
                    //معادل اسمبلی: vfmadd231ps ymm, ymm, ymm
                    //دستور قدرتمند FMA ضرب دو رجیستر اول و جمع با رجیستر سوم در یک سیکل پردازشی
                    //v_sums = (v_gray * v_kernel) + v_sums
                    v_sums = _mm256_fmadd_ps(v_gray, v_kernel, v_sums);
                }
            }

            //حال 8 نتیجه استخراج شده برای هر پیکسل در این رجیستر را در حافظه رم (آرایه) ذخیره میکنم
            //معادل اسمبلی: vmovups [mem], ymm
            float results[8];
            _mm256_storeu_ps(results, v_sums);
            
            //بررسی میکنم که کدام یک از این 8 پنجره بالاترین امتیاز را داشته است
            for(int k = 0; k < 8; k++) {
                float final_score = results[k] / (float)(kWidth * kHeight); //برای عدم وابستگی به سایز کرنل
                if (final_score > max_score) {
                    max_score = final_score;
                    *best_x = x + k; //اینجا باید حواسمون باشه به اضافه k کنیم که ایندکس درست پیکسل رو بیابیم وگرنه همیشه ایندکس پیکسل اول از این 8 تا رو بهمون میده
                    *best_y = y;
                }
            }
        }
    }
    return max_score;
}


void draw_bounding_box(Image *img, int top_x, int top_y, int w, int h) {
    int start_x = top_x, end_x = top_x + w - 1;
    int start_y = top_y, end_y = top_y + h - 1;

    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            // ایجاد یک حاشیه (Border) به ضخامت 2 پیکسل
            if (y == start_y || y == start_y+1 || y == end_y || y == end_y-1 ||
                x == start_x || x == start_x+1 || x == end_x || x == end_x-1) {
                // چک کردن امنیت حافظه برای جلوگیری از خطای Segfault
                if (x >= 0 && x < img->width && y >= 0 && y < img->height) {
                    int idx = (y * img->width + x) * 3;
                    img->data[idx + 0] = 255; // Red (تمام رنگ قرمز)
                    img->data[idx + 1] = 0;   // Green (خاموش)
                    img->data[idx + 2] = 0;   // Blue (خاموش)
                }
            }
        }
    }
}


int main() {
    //خواندن عکس تمپلیت و ایجاد کرنل از روی آن
    int base_w, base_h;
    float* base_kernel = create_kernel_from_image("apple_template.ppm", &base_w, &base_h);
    float threshold = 80.0f; // چون اومدم و اندازه و scale کرنل رو به همان نسبت بزرگ و کوچیک کردم برای جلوگیری از خطاهای احتمالی این استانه تطابق رو روی 80 میگذاریم

    int total_images = 20;           // کل تعداد عکس‌های موجود در دیتاست
    int successful_detections = 0;   // شمارنده تعداد عکس‌هایی که الگو در آن‌ها یافت شد
    double total_c_time_all = 0;     // مجموع زمان اجرای تمام عکس‌ها با الگوریتم C
    double total_avx_time_all = 0;   // مجموع زمان اجرای تمام عکس‌ها با الگوریتم AVX2

    printf("========================================================\n");
    printf("Starting Batch Processing on %d Images...\n", total_images);
    printf("========================================================\n\n");

    // حلقه اصلی برای پیمایش تک تک عکس‌های دیتاست
    for (int i = 1; i <= total_images; i++) {
        
        // تولید مسیر و نام فایل به صورت dataset/image_1.ppm و ...
        char filename[256];
        snprintf(filename, sizeof(filename), "dataset/image_%d.ppm", i);

        // خواندن عکس جاری و تبدیل آن به آرایه اعشاری خاکستری
        Image *image = readPicture(filename);
        float *gray_image = convert_to_grayscale_float(image);

        // متغیرهای زمان‌سنجی و ثبت امتیاز برای "همین عکس جاری"
        float c_best_score = -FLT_MAX;
        double c_total_time = 0;

        float avx_best_score = -FLT_MAX;
        int avx_best_x = -1, avx_best_y = -1, avx_best_w = -1, avx_best_h = -1;
        double avx_total_time = 0;

        // جستجوی الگو در سایزهای مختلف
        for (float scale = 0.5f; scale <= 2.5f; scale += 0.1f) {
            
            int new_w = (int)(base_w * scale);
            int new_h = (int)(base_h * scale);
            
            // رد کردن سایزهای نامعتبر (بزرگتر از عکس یا خیلی کوچک)
            if (new_w < 4 || new_h < 4) continue; 
            if (new_w > image->width || new_h > image->height) continue;

            // تغییر سایز کرنل به مقیاس جاری
            float* current_kernel = resize_kernel(base_kernel, base_w, base_h, new_w, new_h);
            int cur_x, cur_y;
            float cur_score;

            // --- بررسی با C ---
            clock_t start1 = clock();
            cur_score = templateMatch_C(gray_image, image->width, image->height, current_kernel, new_w, new_h, &cur_x, &cur_y);
            clock_t end1 = clock();
            c_total_time += ((double)(end1 - start1)) / CLOCKS_PER_SEC;

            if (cur_score > c_best_score) {
                c_best_score = cur_score;
            }

            // --- بررسی با AVX2 ---
            clock_t start2 = clock();
            cur_score = templateMatch_AVX2(gray_image, image->width, image->height, current_kernel, new_w, new_h, &cur_x, &cur_y);
            clock_t end2 = clock();
            avx_total_time += ((double)(end2 - start2)) / CLOCKS_PER_SEC;

            if (cur_score > avx_best_score) {
                avx_best_score = cur_score;
                avx_best_x = cur_x; avx_best_y = cur_y; 
                avx_best_w = new_w; avx_best_h = new_h;
            }

            free(current_kernel);
        }

        // زمان پردازش این عکس را به زمان پردازش کل عکس ها اضافه میکنم
        total_c_time_all += c_total_time;
        total_avx_time_all += avx_total_time;

        // بررسی نتیجه تشخیص برای عکس جاری و چاپ گزارش خطی
        printf("[Image %02d] C Time: %.4fs | AVX2 Time: %.4fs | ", i, c_total_time, avx_total_time);
        
        if (avx_best_score >= threshold) {
            successful_detections++; // افزایش شمارنده تشخیص‌های موفق
            printf("Status: FOUND! (Score: %.2f) \n", avx_best_score);
            
            // رسم کادر دور لوگو و ذخیره عکس خروجی در همان پوشه
            draw_bounding_box(image, avx_best_x, avx_best_y, avx_best_w, avx_best_h);
            char out_filename[256];
            snprintf(out_filename, sizeof(out_filename), "dataset/output_%d.ppm", i);
            writePicture(image, out_filename);
        } else {
            printf("Status: NOT FOUND (Max Score: %.2f)\n", avx_best_score);
        }

        // آزادسازی حافظه عکس جاری برای جلوگیری از پر شدن حافظه در طول حلقه
        free(gray_image);
        free(image->data);
        free(image);
    }

    printf("\n================ FINAL REPORT ======================\n");
    printf("Total Images Processed : %d\n", total_images);
    printf("Successful Detections  : %d out of %d (Accuracy: %.1f%%)\n", 
           successful_detections, total_images, 
           ((float)successful_detections / total_images) * 100.0f);
    
    printf("\n--- Performance Benchmarking ---\n");
    printf("Total Time (Standard C) : %.4f seconds\n", total_c_time_all);
    printf("Total Time (AVX2 FMA)   : %.4f seconds\n", total_avx_time_all);
    
    if (total_avx_time_all > 0) {
        printf("Overall Speedup         : %.2f x Faster using AVX2\n", 
               total_c_time_all / total_avx_time_all);
    }
    printf("========================================================\n");

    free(base_kernel);

    return 0;
}