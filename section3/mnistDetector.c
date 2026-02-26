#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h> // کتابخانه حیاتی برای استفاده از دستورات سیمد و رجیسترهای 256 بیتی AVX2

// کتابخانه معروف stb برای خواندن آسان عکس‌ها از روی هارد دیسک
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" 

// فایلی که شامل آرایه‌های وزن‌ها و بایاس‌های استخراج شده از مدل آموزش‌دیده (مثلاً در PyTorch) است.
#include "weights.h"


// تابع جمع افقی: تبدیل 8 عدد داخل یک رجیستر AVX به یک عدد float معمولی
// در این تابع میخواهیم 8 عدد فلوت موجود در یک رجیستر 256 بیتی را با هم چمع کرده و یک عدد کنیم
float hsum_avx(__m256 v) {
    //با توجه به این که دستورات موازی در avx مخصوص عملیات های عمودی بین دو عدد متناظر در دو رجیستر هستند و نه عملیات روی اعداد موجود در یک رجیستر
    // میایم و این کار رو به شکل زیر انجام میدیم
    // [a , b, c, d ,e , g , f , h]
    __m128 vlow  = _mm256_castps256_ps128(v);      // گرفتن 4 عدد پایینی [a , b , c, d] با 4 عدد پایین این رجیستر کاری میکنه که مثل یه رجیستر 128 بیتی مستقل رفتار کنه
    __m128 vhigh = _mm256_extractf128_ps(v, 1);    // گرفتن 4 عدد بالایی [e , f , g, h] معادل اسمبلی: vextractf128 xmm, ymm, 1
    // جمع کردن نیمه بالا و پایین (کاهش 8 عدد به 4 عدد)
    vlow  = _mm_add_ps(vlow, vhigh); // [w , x , y ,z] 
    
    // کپی کردن و جابجا کردن 2 عدد بالایی به 2 جایگاه پایینی
    __m128 shuf  = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1)); // [y , z , y , z] معادل اسمبلی: vshufps xmm, xmm, xmm, imm8 بر زدن اعداد داخل رجیستر
    
    // vlow , shuf جمع کردن
    vlow  = _mm_add_ps(vlow, shuf); // [y+w , z+x , 2y , 2z]
    
    // نیمه بالایی رجیستر shuf رو بر میداره و در نیمه پایین رجیستر vlow کپی میکنه
    shuf  = _mm_movehl_ps(shuf, vlow); // [z+x , ....] معادل اسمبلی: vmovhlps xmm, xmm, xmm
    
    // جمع نهایی (رسیدن به جمع کل در خانه ایندکس صفر)
    vlow  = _mm_add_ps(vlow, shuf); // [y+w+z+x]
    
    // استخراج عدد نهایی از رجیستر SIMD به یک متغیر استاندارد C
    //معادل اسمبلی: vmovss reg, xmm
    return _mm_cvtss_f32(vlow);
}

// تابع بارگذاری و پیش‌پردازش عکس (تبدیل به فرمتی که شبکه عصبی می‌فهمد)
void load_and_preprocess_image(const char* filename, float output[28][28]) {
    int w, h, channels;
    // عدد رو لود میکنیم با این تابع از کتابخانه stb و در ورودی آخر این تابع هم میگیم که تک کاناله یعنی سیاه و سفید آن را بخواند
    unsigned char* img = stbi_load(filename, &w, &h, &channels, 1); 
    
    if (img == NULL) {
        printf("Error: Could not load image %s\n", filename);
        exit(1);
    }

    if (w != 28 || h != 28) { // با توجه یبه اینکه دیتا ستی که باهاش مدل رو train کردیم عکس های از اعداد mnist 28 x 28 بوده اند باید عکس ورودی مون هم در همین ابعاد باشه
        printf("Warning: Image size is %dx%d (expected 28x28). Using crop/resize logic is recommended.\n", w, h);
    }

    // استانداردسازی (Z-Score Normalization):
    // شبکه عصبی اعدادی بین 0 تا 255 را دوست ندارد و باید حول صفر با واریانس مشخص پخش شوند اعداد
    // اعداد 0.1307 (میانگین) و 0.3081 (انحراف معیار) مختص دیتاسیت MNIST هستند.
    // در خود مدل هم اعداد پیکسل را به این صورت نرمالایز میکنیم : (pixel - mean) / std
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float pixel_val = (float)img[y * w + x] / 255.0f; // ابتدا تبدیل به عددی بین 0 تا 1
            output[y][x] = (pixel_val - 0.1307f) / 0.3081f;   // سپس اعمال فرمول (X - Mean) / Std
        }
    }

    stbi_image_free(img); 
}

// نسخه 1: پیاده‌سازی معمولی با C
// به طور کلی ما 4 مرحله و لایه داریم در این شبکه cnn
int predict_mnist_naive(float image[28][28]) {
    // آرایه 4 بعدی: 4 فیلتر، هر کدام خروجی 26 در 26 می‌دهند
    float conv_out[4][26][26] = {0};
    
    //  لایه اول convolution 
    // ابتدا عکس را با 4 فیلتر کانولوشن میکنیم
    // به این لایه، لایه features extractor هم میگن
    for(int f=0; f<4; f++) {             // حرکت روی 4 فیلتر مختلف (استخراج 4 ویژگی مختلف از عکس)
        for(int y=0; y<26; y++) {        // حرکت عمودی روی عکس
            for(int x=0; x<26; x++) {    // حرکت افقی روی عکس
                float sum = conv_bias[f]; // شروع جمع با مقدار بایاس مربوط به این فیلتر
                
                // اعمال فیلتر 3 در 3 روی ناحیه فعلی
                for(int ky=0; ky<3; ky++) {
                    for(int kx=0; kx<3; kx++) {
                        sum += image[y+ky][x+kx] * conv_weights[f][ky][kx]; // کانولوشن ایندکس متناظر از هر کدام از فیلتر ها روی پنجره باز شده
                    }
                }
                conv_out[f][y][x] = sum; // ایجاد 4 عکس 26 در 26 جدید از روی عکس ورودی
            }
        }
    }

    //  لایه‌ی دومReLU و MaxPool و Flatten ---
    float flattened[676]; // 4 * 13 * 13 = 676 (تبدیل آرایه سه بعدی به یک بردار خطی یک بعدی)
    int idx = 0;
    
    for(int f=0; f<4; f++) {
        for(int y=0; y<26; y+=2) {       // ابتدا برای بخش maxpool باید بیایم و بین هر پنجره 4 تایی ماکسیممش رو انتخاب کنیم و در عکس جدید بگذاریم
            for(int x=0; x<26; x+=2) {   // این مر مرحله maxpool عملا میاد و بخش هایی با ویژگی برجسته تر رو استخراج میکنه
                float max_val = -1e9;    // مقداردهی اولیه با یک عدد خیلی منفی
                
                // بررسی 4 پیکسل (در یک پنجره 2 در 2) برای یافتن بزرگترین عدد
                for(int ky=0; ky<2; ky++) {
                    for(int kx=0; kx<2; kx++) {
                        float val = conv_out[f][y+ky][x+kx];
                        
                        // تابع فعال‌ساز ReLU: اگر عدد منفی بود، صفرش کن
                        if (val < 0) val = 0; // حذف بیت هایی که وِیژگی مورد نظر را ندارند
                        
                        // عملیات MaxPool: ذخیره بزرگترین عدد
                        if (val > max_val) max_val = val;
                    }
                }
                flattened[idx++] = max_val; // ذخیره متوالی در آرایه 1 بعدی
            }
        }
    }

    //  لایه سوم Fully Connected (تولید 10 امتیاز برای اعداد 0 تا 9) 
    float scores[10]; // در این مرحله ما باید از روی آرایه خطی حاوی تمام پیکسل ها خروجی را مشخص کنیم
    // بین هر کدام از این پیکسل های استخراج شده و اعداد 0 تا 9 یک ارتباطی وزنی و بایاسی هست که از مدل ترین شده استخراج کرده ایم و در فایل weights.h قرار دارد
    for(int i=0; i<10; i++) { // محاسبه امتیاز برای هر کلاس (0 تا 9)
        float sum = fc_bias[i]; // لود مقدار بایاس برای هر خروجی
        for(int j=0; j<676; j++) {
            sum += flattened[j] * fc_weights[i][j]; // ضرب برداری ورودی در وزن‌های این کلاس
        }
        scores[i] = sum; // امتیاز نهایی محاسبه شده برای اینکه عدد ورودی چقدر شبیه به هر کدام از اعداد 0 تا 9 است
    }

    // یافتن بیشترین امتیاز  مرحله 4
    int best = 0;
    for(int i=1; i<10; i++) {
        if(scores[i] > scores[best]) best = i; // ایندکس کلاسی که بیشترین امتیاز را دارد برگردان
    }
    return best;
}

// نسخه 2: پیاده‌سازی بهینه با AVX2 
int predict_mnist_avx(float image[28][28]) {
    
    float conv_out[4][26][28] = {0}; // شبیه به توضیحات در پیاده سازی نسخه اول
    

    // لایه اول : کانولوشن (Convolution Layer) با AVX2
    for (int f = 0; f < 4; f++) { 
        for (int y = 0; y < 26; y++) {
            
            int x = 0;
            // ما 26 پیکسل در عرض داریم و  با گام 8تایی جلو می‌رویم
            for (; x <= 26 - 8; x += 8) {
                
                // مقدار بایاس این فیلتر را کپی میکنیم توی 8 خانه یک رجیستر تا برای جمع آماده باشد
                __m256 sum_vec = _mm256_set1_ps(conv_bias[f]);

                // اعمال کرنل 3x3
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        
                        // به جای خواندن یک پیکسل، 8 پیکسل متوالی را همزمان میخوانیم
                        __m256 img_pixels = _mm256_loadu_ps(&image[y + ky][x + kx]);
                        
                        // مقدار یک ایندکس از کرنل (وزن) را بر میدارم و در تمام 8 خانه یک رجیستر کپی میکنم.
                        // این یک وزن ثابت است که باید در آن 8 پیکسل ضرب شود
                        __m256 weight = _mm256_set1_ps(conv_weights[f][ky][kx]);
                        
                        // در یک کلاک سایکل، هم ضرب می‌کند و هم با مقدار قبلی جمع می‌کند.
                        // sum_vec = (img_pixels * weight) + sum_vec
                        sum_vec = _mm256_fmadd_ps(img_pixels, weight, sum_vec);
                    }
                }
                // 8 خروجی نهایی را که با هم محاسبه شده‌اند، به یکباره در آرایه ذخیره کن
                _mm256_storeu_ps(&conv_out[f][y][x], sum_vec);
            }
            
            // برای پیکسل‌های 24 و 25 که تعدادشان به 8 نرسید جداگانه حساب میکنیم
            // اینها را به روش قدیمی اسکالر حساب می‌کنیم تا از مرز تصویر بیرون نزنیم
            for (; x < 26; x++) {
                float sum = conv_bias[f];
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        sum += image[y + ky][x + kx] * conv_weights[f][ky][kx];
                    }
                }
                conv_out[f][y][x] = sum;
            }
        }
    }

    // لایه دوم پولینگ (MaxPool) و ReLU و تبدیل به بردار (Flatten)
    float flattened[676]; 
    int flat_idx = 0;

    // توضیحات به مانند نسخه اول
    for (int f = 0; f < 4; f++) {
        for (int y = 0; y < 26; y += 2) { // پنجره های 2 در 2 و انتخاب بزرگترین پسکل از هر یک  (MaxPool)
            for (int x = 0; x < 26; x += 2) {
                float max_val = -10000.0f; 
                
                for (int ky = 0; ky < 2; ky++) {
                    for (int kx = 0; kx < 2; kx++) {
                        float val = conv_out[f][y + ky][x + kx];
                        
                        if (val < 0.0f) val = 0.0f; // ReLU
                        if (val > max_val) max_val = val; // MaxPool
                    }
                }
                flattened[flat_idx++] = max_val;
            }
        }
    }

    // لایه سوم (Fully Connected) با AVX2
    float final_scores[10];

    for (int i = 0; i < 10; i++) { // برای هر رقم از 0 تا 9
        // ساخت یک رجیستر که تمام 8 خانه‌اش صفر است (برای نگه داشتن مجموع‌های جزئی)
        __m256 sum_vec = _mm256_setzero_ps();
        
        int j = 0;
        // ما 676 ویژگی داریم. 8 تا 8 تا جلو می‌رویم تا ضرب نقطه‌ای انجام دهیم.
        for (; j <= 676 - 8; j += 8) { // 4 * 13 * 13
            // خواندن 8 فیچر (خروجی‌های Flatten شده)
            __m256 feat = _mm256_loadu_ps(&flattened[j]);
            
            // خواندن 8 وزن متناظر با این فیچرها برای عدد iام
            __m256 w = _mm256_loadu_ps(&fc_weights[i][j]);
            
            // ضرب دو به دوی فیچرها در وزن‌ها و جمع با مجموع‌های قبلی
            sum_vec = _mm256_fmadd_ps(feat, w, sum_vec);
        }
        
        // در اینجا ما یک رجیستر داریم که حاوی 8 عدد است
        // باید این 8 عدد را با هم جمع کنیم تا به یک امتیاز کلی برسیم.
        float total = hsum_avx(sum_vec);
        
        // 676 بر 8 بخش‌پذیر نیست (676 = 84 * 8 + 4). 
        // 4 ویژگی نهایی را به صورت معمولی بررسی میکنیم
        for (; j < 676; j++) {
            total += flattened[j] * fc_weights[i][j];
        }
        
        // نهایتاً بایاس را اضافه کرده و امتیاز این کلاس را ثبت می‌کنیم.
        final_scores[i] = total + fc_bias[i];
    }

    // لایه نهایی: یافتن بیشترین امتیاز
    int best_class = 0;
    float max_score = final_scores[0];
    for (int i = 1; i < 10; i++) {
        if (final_scores[i] > max_score) {
            max_score = final_scores[i];
            best_class = i; // عدد تشخیص داده شده توسط شبکه را به‌روزرسانی میکنم
        }
    }

    return best_class;
}

int main() {
    const char* img_path = "test_image.png"; 
    float input_image[28][28];
    
    printf("Loading image: %s ...\n", img_path);
    load_and_preprocess_image(img_path, input_image);

    // زمان‌گیری از نسخه معمولی برای اینکه در ابعاد ثاینه باشد باید چنیدین با احراش کنیم با آزمون خطا اردر 10000 براش خوبه
    printf("Running Naive Implementation (10,000 runs)...\n");
    clock_t start_naive = clock();
    int pred_naive = 0;
    for(int i=0; i<10000; i++) {
        pred_naive = predict_mnist_naive(input_image);
    }
    clock_t end_naive = clock();
    double time_naive = (double)(end_naive - start_naive) / CLOCKS_PER_SEC;

    // زمان‌گیری از نسخه AVX2
    printf("Running AVX2 Implementation (10,000 runs)...\n");
    clock_t start_avx = clock();
    int pred_avx = 0;
    for(int i=0; i<10000; i++) {
        pred_avx = predict_mnist_avx(input_image);
    }
    clock_t end_avx = clock();
    double time_avx = (double)(end_avx - start_avx) / CLOCKS_PER_SEC;

    // نمایش گزارش عملکرد
    printf("\n================ RESULTS ================\n");
    printf("Prediction (Naive): %d\n", pred_naive);
    printf("Prediction (AVX2) : %d\n", pred_avx);
    printf("-----------------------------------------\n");
    printf("Time Naive : %.4f seconds\n", time_naive);
    printf("Time AVX2  : %.4f seconds\n", time_avx);
    
    if (time_avx > 0) {
        printf("Speedup    : %.2fx faster!\n", time_naive / time_avx); // محاسبه میزان افزایش سرعت
    }
    printf("=========================================\n");

    return 0;
}
