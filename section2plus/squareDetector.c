#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>

// حد آستانه برای تشخیص لبه. مقادیر گرادیان بالاتر از این عدد، لبه محسوب می‌شه
#define EDGE_THRESHOLD 150.0f 
// حداقل طول یک ضلع (به پیکسل) برای اینکه یک شکل به عنوان مربع بررسی بشه
#define MIN_SQUARE_SIZE 20
// یک درصد برای اینکه چند درصد از اضلاع مربع باید لبه باشند تا شکل مربع تشخیص داده شود برای کاهش سخت گیری در تشخیص
#define ACCEPTANCE_RATIO 0.75f 
// تعداد کل عکس‌های موجود در پوشه دیتاست 
#define DATASET_SIZE 100

// با توجه به اینکه طبق ایده کد من فقط در بخش اعمال کرنل سوبل از اسمبلی استفاده میشه و بخش عظیم زمان صرف بررسی و
// کشف مربع میشود من این عملیات اعمال فیلتر را 100 بار تگرارامیکنم تا اردر زمانی آن به ms که تابع clock از ان استفاده میکند برسد
#define BENCHMARK_ITERATIONS 100 

// ساختار نگهداری اطلاعات تصویر
typedef struct {
    int width;
    int height;
    unsigned char *data;
} Image;

// ساختار نگهداری مختصات و مشخصات مربع پیدا شده
typedef struct {
    int x;       // مختصات X گوشه بالا-چپ
    int y;       // مختصات Y گوشه بالا-چپ
    int size;    // طول ضلع بالای پیدا شده (عرض شکل)
    float score; // امتیاز تطابق (چند درصد از اضلاع کامل بودند)
    int found;   // فلگ وضعیت (1 یعنی پیدا شده، 0 یعنی پیدا نشده)
} SquareResult;

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


// اعمال سوبل به روش کد های c
void apply_sobel_C(const float* gray, float* edges, int w, int h) {
    // با توجه به اینکه کرنل سوبل اعمال شده رو حالت 3 در 3 معمولی گرفتم یعنی
    // [ -1 0 1
    //   -2 0 2
    //   -1 0 1 ] برای GX

    // [ -1 -2 -1
    //   0 0 0
    //   1 2 1 ] برای GY
    //برای اینکه در  گرفتن پیکسل های توصیر اصلی به عنوان پنجره از محدوده خارج نشوم تا یک پیکسل به آخر رو سلکت میکنم
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            // محاسبه گرادیان افقی (تغییرات رنگ در راستای محور X)
            float gx = -gray[(y-1)*w + (x-1)] + gray[(y-1)*w + (x+1)]
                       -2.0f*gray[y*w + (x-1)] + 2.0f*gray[y*w + (x+1)]
                       -gray[(y+1)*w + (x-1)] + gray[(y+1)*w + (x+1)];
            
            // محاسبه گرادیان عمودی (تغییرات رنگ در راستای محور Y)
            float gy = -gray[(y-1)*w + (x-1)] - 2.0f*gray[(y-1)*w + x] - gray[(y-1)*w + (x+1)]
                       +gray[(y+1)*w + (x-1)] + 2.0f*gray[(y+1)*w + x] + gray[(y+1)*w + (x+1)];
            
            // برای سرعت بیشتر در مجاسبه به جای مقدار اندازه اقلیدسی فاصله منهتن شون رو با دو تابع abs محاسبه میکنم
            edges[y * w + x] = fabsf(gx) + fabsf(gy);
        }
    }
}

// اعمال برداری و موازی فیلتر سوبل با دستورات AVX2 
void apply_sobel_AVX2(const float* gray, float* edges, int w, int h) {
    // ایجاد یک ماسک بیتی برای حذف بیت علامت برای ایجاد قدرت مطلق اعداد اعشاری
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    // وکتور شامل 8 عدد 2.0 برای ضرب‌های وزن‌دار ماتریس سوبل
    __m256 two = _mm256_set1_ps(2.0f);

    for (int y = 1; y < h - 1; y++) {
        // هر 8 عدد مجاور را در یک مرحله لود و برای هر کدام در پنجره مختص به خودش کرنل را مانولوشن میکنم
        for (int x = 1; x <= w - 1 - 8; x += 8) {
            // هر 8 خانه مجاور پیکسل مورد نظر را با شروع از آدرس بیس که مربوط به اولین پیکسل این 8 پیکسل سلکت شده است انتخاب میکنم
            __m256 top_left  = _mm256_loadu_ps(&gray[(y-1)*w + (x-1)]); // 8 عدد اعشاری را انتخاب میکند با شروع از  top left پیکسل اول 
            __m256 top_mid   = _mm256_loadu_ps(&gray[(y-1)*w + x]); // 8 عدد اعشاری را انتخاب میکند با شروع از  top mid پیکسل اول  
            __m256 top_right = _mm256_loadu_ps(&gray[(y-1)*w + (x+1)]);// 8 عدد اعشاری را انتخاب میکند با شروع از  top right پیکسل اول 
            
            __m256 mid_left  = _mm256_loadu_ps(&gray[y*w + (x-1)]);// 8 عدد اعشاری را انتخاب میکند با شروع از  mid left پیکسل اول 
            __m256 mid_right = _mm256_loadu_ps(&gray[y*w + (x+1)]);// 8 عدد اعشاری را انتخاب میکند با شروع از  mid right پیکسل اول 
            
            __m256 bot_left  = _mm256_loadu_ps(&gray[(y+1)*w + (x-1)]);// 8 عدد اعشاری را انتخاب میکند با شروع از  bottom left پیکسل اول 
            __m256 bot_mid   = _mm256_loadu_ps(&gray[(y+1)*w + x]);// 8 عدد اعشاری را انتخاب میکند با شروع از  bottom mid پیکسل اول 
            __m256 bot_right = _mm256_loadu_ps(&gray[(y+1)*w + (x+1)]);// 8 عدد اعشاری را انتخاب میکند با شروع از  bottom right پیکسل اول 

            // محاسبه ی gx با توجه به کرنل سوبل
            __m256 gx = _mm256_sub_ps(top_right, top_left);
            __m256 mid_diff = _mm256_sub_ps(mid_right, mid_left);
            // استفاده از دستور fmadd برای برای انجام ضرب mid diff در 2 و جمع با خودش طبق چیزی که در بخش کد c توضیح دادم
            gx = _mm256_fmadd_ps(two, mid_diff, gx); // gx = (mid_diff * 2.0) + gx
            gx = _mm256_add_ps(gx, _mm256_sub_ps(bot_right, bot_left));

            // محاسبه Gy
            __m256 top_sum = _mm256_add_ps(top_left, top_right); // دوباره طبق همان کدی که برای بخش c نوشتم باید راست بالا و راست چپ ضرایب یک داشته باشند و با هم جمع میشوند
            top_sum = _mm256_fmadd_ps(two, top_mid, top_sum); // اندیس وسط بالا ضریب 2 میگیرد پس آن را در دو ضراب و با مجموع دو اندیس بالای دگیر جمع میکنیم
            
            __m256 bot_sum = _mm256_add_ps(bot_left, bot_right); // این دو اندیس پایین چپ و پایین راست هم ضرایت -1 دارند و باید با هم جمعشان کرده و پس از جمع با 2 برابر اندیس پایین وسط آن را از کل کم کرد
            bot_sum = _mm256_fmadd_ps(two, bot_mid, bot_sum); // اندیس وسط پایین ضریب -2 میگیرد و باید در 2 آن را ضرب گرد و با مجموع دو اندیس دیگر پایین جمع و در نهایت از کل کم کرد
            __m256 gy = _mm256_sub_ps(bot_sum, top_sum);

            // محاسبه قدر مطلق با استفاده از ماسک بیتی و And گرفتن
            __m256 abs_gx = _mm256_and_ps(gx, sign_mask);
            __m256 abs_gy = _mm256_and_ps(gy, sign_mask);
            
            // جمع بردارهای قدر مطلق Gx و Gy
            __m256 mag = _mm256_add_ps(abs_gx, abs_gy);

            // ذخیره همزمان 8 نتیجه در آرایه خروجی
            _mm256_storeu_ps(&edges[y * w + x], mag);
        }
    }
}

SquareResult detect_dynamic_square(const float *sobelImage, int w, int h) {
    SquareResult best = {0, 0, 0, 0.0f, 0};
    int MAX_GAP_COUNT = 5;
    for (int y = 0; y < h - MIN_SQUARE_SIZE; y++) {
        for (int x = 0; x < w - MIN_SQUARE_SIZE; x++) {
            
            // پیدا کردن اولین نقطه ای که به اندازه کافی تیز باشه به عنوان لبه
            // این نقطه به عنوان گوشه چپ بالای مربع مورد نظر خواهم گرفت چون پیمایش  ما روی پیکسل های مربع از چپ به راست و از بالا به پایین است
            // در ادامه تا با آزمایش و خط فهمیدم که کد بازدهی پایینی دارد اگر وجو پیکسل های گپ در ضلع بالایی را در نظر نگیریم و این مشکل ختی با پایین ارودن EDGE THRESHOLD هم درست نمیشود
            if (sobelImage[y * w + x] >= EDGE_THRESHOLD) {
                int size = 1;
                int gap_count = 0; // متغیری برای شمارش تعداد گپ های پیوسته
                // تا جایی حلقه را ادامه میدهیم که از تصویز اسلی بیرون نزنیم
                while (x + size < w) {
                    if (sobelImage[y * w + x+size] > EDGE_THRESHOLD)
                    {
                        size++; // اندازه ضلع را زیاد کن
                        gap_count = 0; // این متغیر شمارش گپ های پیوسته رو ریست کن
                    }
                    else {
                        gap_count++;
                        if (gap_count > MAX_GAP_COUNT) {
                            break; // اگر گپ خیلی طولانی شد، خط واقعا تزمام شده 
                        }
                        size++;
                    }
                }

                size -= gap_count; // گپ های انتهایی را که الکی شماردیم تا مطمئن بشیم تمام شده حذف میکنم

                // بررسی اینکه آیا خط افقی پیدا شده حداقل سایز یک ضلع را دارد و از عکس  اصلی بیرون نمی‌زند
                if (size >= MIN_SQUARE_SIZE && y + size < h) {
                    
                    // محاسبه تعداد کل پیکسل‌هایی که در سه ضلع دیگر انتظار دارم به باشند
                    // (size-1) برای ضلع چپ، (size-1) برای ضلع راست، و (size+1) برای طول ضلع پایین
                    int expected_edge_pixels = (size - 1) * 2 + (size + 1); 
                    int actual_edge_pixels = 0;

                    // در یه حلقه میام و به صورت موازی دو ضلع عمودی را بررسی میکنم
                    for (int dy = 1; dy < size; dy++) {
                        // بررسی لبه در ضلع چپ
                        if (sobelImage[(y + dy) * w + x] >= EDGE_THRESHOLD) 
                            actual_edge_pixels++;
                        // بررسی لبه در ضلع راست
                        if (x + size < w && sobelImage[(y + dy) * w + (x + size)] >= EDGE_THRESHOLD) 
                            actual_edge_pixels++;
                    }

                    // اسکن کردن خط افقی پایین (ضلع زیرین)
                    for (int dx = 0; dx <= size; dx++) {
                        if (y + size < h && sobelImage[(y + size) * w + (x + dx)] >= EDGE_THRESHOLD) 
                            actual_edge_pixels++;
                    }

                    // محاسبه امتیاز شکل (نسبت پیکسل‌های لبه واقعی به مورد انتظار)
                    float score = (float) actual_edge_pixels / expected_edge_pixels;
                    
                    // این خطو اضافه کردم برای اینکه ممکنه کپ هایی  اضافه در این مسیر شمرده شده باشند
                    if (score > 1.0f) score = 1.0f;

                    // ثبت نتیجه در صورت پاس کردن حد آستانه و داشتن امتیاز بهتر از شکل‌های قبلی که پیدا کرده بودم
                    if (score >= ACCEPTANCE_RATIO && score > best.score) {
                        best.x = x;
                        best.y = y;
                        best.size = size;
                        best.score = score;
                        best.found = 1;
                    }
                }
            }
        }
    }
    return best;
}


void draw_bounding_box(Image *img, int top_x, int top_y, int size) {
    for (int y = top_y; y <= top_y + size; y++) {
        for (int x = top_x; x <= top_x + size; x++) {
            // رسم یک کادر با ضخامت 2 پیکسل حول اون مربعی که اطلاعاتشو سیو کردم
            if (y == top_y || y == top_y+1 || y == top_y+size || y == top_y+size-1 ||
                x == top_x || x == top_x+1 || x == top_x+size || x == top_x+size-1) {
                // بررسی برای اینکه بردر از تصویر نطنه بیرون که به segfault بخورم
                if (x >= 0 && x < img->width && y >= 0 && y < img->height) {
                    int idx = (y * img->width + x) * 3;
                    img->data[idx] = 255;     // قرمز
                    img->data[idx+1] = 0; // سبز
                    img->data[idx+2] = 0;   // آبی
                }
            }
        }
    }
}


int main() {
    printf("========== Project: Dynamic Geometric Shape Detection ==========\n");
    printf("Processing %d images from 'dataset' folder...\n", DATASET_SIZE);
    printf("Benchmarking Sobel filter over %d iterations per image...\n\n", BENCHMARK_ITERATIONS);
 
    int total_processed = 0; // مجموع کل زمان c , avx
    int squares_found = 0; // تعداد مربع های یافت شده
    double total_time_c = 0.0; 
    double total_time_avx = 0.0;

    char in_filename[256]; // نام فایل ورودی
    char out_filename[256]; // نام فایل تصویر خروجی

    // حلقه پردازش تک تک عکس‌های دیتاست
    for (int i = 1; i <= DATASET_SIZE; i++) {
        sprintf(in_filename, "dataset/image_%d.ppm", i);
        
        Image *image = readPicture(in_filename); // خواندن به ترتیب عکس ها
        if (!image){ 
            continue; // اگر عکس وجود نداشت، برو عکس بعدی
        }

        total_processed++;
        float *gray_image = convert_to_grayscale_float(image);
        
        // اختصاص حافظه صفر شده (calloc) بیرون از تایمر برای جلوگیری از دخالت در زمان‌سنجی
        float *sobel_c = (float*)calloc(image->width * image->height, sizeof(float));
        float *sobel_avx = (float*)calloc(image->width * image->height, sizeof(float));

        // پردازش و زمان‌بندی مسیر C
        clock_t t1 = clock();
        for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
            apply_sobel_C(gray_image, sobel_c, image->width, image->height);
        }
        clock_t t2 = clock();
        double time_c_ms = (((double)(t2 - t1)) / CLOCKS_PER_SEC * 1000.0) / BENCHMARK_ITERATIONS; 

        //پردازش و زمان‌بندی مسیر AVX2 
        clock_t t3 = clock();
        for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
            apply_sobel_AVX2(gray_image, sobel_avx, image->width, image->height);
        }
        clock_t t4 = clock();
        double time_avx_ms = (((double)(t4 - t3)) / CLOCKS_PER_SEC * 1000.0) / BENCHMARK_ITERATIONS; 

        // اجرای این الگوریتم پیدا کردن مربع از خروجی کانولوشن داده شده با کرنل سوبل را خارح از تایمر بررسی میکنم زیرا نسبت به بخش های که از avx استفاده کردم تایم زیادی را به خود اختصاص میدهد و به چون دستورات simd ندارد عملا نشان دهنده ی نسبت سرعت این دستورات به دستورات c نحواهد بود
        SquareResult result_avx = detect_dynamic_square(sobel_avx, image->width, image->height);

        total_time_c += (time_c_ms / 1000.0);
        total_time_avx += (time_avx_ms / 1000.0);
        // گزارش شماره عکس و امتیاز  و سایز و وضعیت وجود یا عدم وجود مربع و تایم بررسی با c و بررسی با avx برای هر عکس ورودی
        printf("Image #%03d | Score: %6.2f%% | Size: %3dx%3d | Status: %s | Sobel C: %7.4f ms | Sobel AVX2: %7.4f ms\n", 
               i, 
               result_avx.score * 100.0f, 
               result_avx.size, result_avx.size,
               result_avx.found ? "[FOUND]  " : "[NOT FOUND]", 
               time_c_ms, time_avx_ms);

        // اگر شکل پیدا شد، ذخیره در خروجی
        if (result_avx.found) {
            squares_found++;
            draw_bounding_box(image, result_avx.x, result_avx.y, result_avx.size);
            sprintf(out_filename, "output/result_%d.ppm", i);
            writePicture(image, out_filename);
        }

        // آزادسازی حافظه رم هر عکس در پایان حلقه (جلوگیری از Memory Leak)
        free(gray_image);
        free(sobel_c);
        free(sobel_avx);
        free(image->data);
        free(image);
    }

    // چاپ گزارش آماری نهایی
    printf("\n==================== FINAL REPORT ====================\n");
    printf("Total Images Processed : %d\n", total_processed);
    printf("Total Squares Found    : %d out of %d\n", squares_found, total_processed);
    printf("Total Pure Sobel Time C: %.5f seconds\n", total_time_c);
    printf("Total Pure Sobel AVX2  : %.5f seconds\n", total_time_avx);
        
    double speedup = total_time_avx > 0 ? (total_time_c / total_time_avx) : 0.0;
    printf("Average Speedup (AVX2 over C): %.2f X\n", speedup);
        
    
    printf("======================================================\n");

    return 0;
}
