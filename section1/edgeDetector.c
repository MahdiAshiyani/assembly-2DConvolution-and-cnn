#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ساختار داده‌ای تصویر
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

unsigned char clamp(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (unsigned char)value;
}

void convolution2D(Image *outImage, Image *inImage, float *kernel, int kSize)
{
    int height = inImage->height;
    int width = inImage->width;
    int offset = kSize / 2;
    for (int i = offset; i < inImage->height - offset; i++) // در پیاده سازی کانولوشن با c مجبوریم به صورت sequential برای هر پیکسل 9 بار عمل ضرب و جمع متوالی انجام دهیم
    {
        for (int j = offset; j < inImage->width - offset; j++)
        {
            int rSum = 0; // متغیری برای نگه داری مقدار بایت رنگ قرمز نهایی پیکسل بعد کانولوشن
            int gSum = 0; // متغیری برای نگه داری مقدار بایت رنگ سبز نهایی پیکسل بعد کانولوشن
            int bSum = 0; // متغیری برای نگه داری مقدار بایت رنگ آبی نهایی پیکسل بعد کانولوشن

            for (int m = -offset; m <= offset; m++)
            {
                for (int n = -offset; n <= offset; n++) // باید در یک پنجره ی 3 در 3 برای هر  پیکسل ، پیکسل همسایه نطیر آن را با پیکسل موجود در کرنل ضرب کنیم برای هر رنگ و سپس با مجموع قبلی ها جمع کنیم
                {
                    int pixelIndex = ((i + m) * width + (j + n)) * 3; // محاسبه مکان پیکسل متناظر در عکس اصلی به عنوان بایت اول رنگ اون پیکسل
                    rSum += kernel[(m + offset) * kSize + (n + offset)] * inImage->data[pixelIndex + 0]; // با توجه به اینکه هر پیکسل سه بایت را به ترتیب برای ذخیره رنگ قرمز و سبز و آبی برای خود رزرو کرده این بایت برای رنگ قرمز آن پیکسل است
                    gSum += kernel[(m + offset) * kSize + (n + offset)] * inImage->data[pixelIndex + 1]; // ضرب پیکسل متانظر کدنل در رنگ سبر پیکسل متناظر در عکس ورودی
                    bSum += kernel[(m + offset) * kSize + (n + offset)] * inImage->data[pixelIndex + 2]; // ضرب پیکسل متانظر کدنل در رنگ آبی پیکسل متناظر در عکس ورودی
                }
            }

            int outIndex = (i * width + j) * 3; // ضریب 3 با توجه به اینکه برای هر پیکسل  3 بایت اشغال میشه
            outImage->data[outIndex + 0] = clamp(rSum); // اگه مقدار عددی هر رنگ برای هر پیکسل از 255 بیشتر بشه باید همون 255 در نظر بگیریم
            outImage->data[outIndex + 1] = clamp(gSum);
            outImage->data[outIndex + 2] = clamp(bSum);
        }
    }
}

void convolution2D_Intrinsic(Image *outImage, Image *inImage, float *kernel, int kSize)
{
    int channels = 3; // تعداد بایت ها (رنگها) برای هر پیکسل
    int width = inImage->width;
    int height = inImage->height;

    // تبدیل بایت ها از unsigned char به float در یه حافظه ی موقت
    float *floatData = (float *)malloc(width * height * channels * sizeof(float));
    for (int i = 0; i < width * height * channels; i++) {
        floatData[i] = (float)inImage->data[i];
    }

// ما در این روش گام های 8 پیکسلی بر میداریم و هر بار به صورت همزمان عمل کانولوشن را برای 8 پیکسل مجاور شر.ع به انجام میکنیم
//     // در این روش ما از دستور gather استفاده میکنیم تا بتوانیم به بایت های غیر متوالی از عکس دسترسی پیدا کنیم تا بتوانیم هر با برای هر رنگ این عملیات را برای هر پسکسل در 32 بیت مختص به خودش انجام دهیم
//     // این بردار مشخص می‌کند برای برداشتن 8 پیکسل متوالی از یک کانال خاص (مثلاً R)،
//     // چقدر باید در آرایه به جلو بپریم (3 تا 3 تا).
//     // مقادیر: [0, 3, 6, 9, 12, 15, 18, 21]
    __m256i INDEX = _mm256_setr_epi32(0, channels, 2 * channels, 3 * channels, 4 * channels, 5 * channels, 6 * channels, 7 * channels);
//     // پیمایش سطره از 1 تا یکی مانده به آخر (به خاطر کرنل 3x3 و جلوگیری از خروج از مرز)
    for(int i = 1; i < height - 1; i++) {
        //         //  8 پیکسل در هر مرحله پردازش می‌شود
        for(int j = 1; j < width - 8; j += 8) {
            //             // پردازش کاملاً مستقل و مجزای هر کانال رنگی (R, G, B)
            for(int c = 0; c < channels; c++) {
                
                //معادل اسمبلی: vxorps ymm, ymm, ymm (صفر کردن رجیستر 256 بیتی)
                // رجیستر جمع کننده را برای 8 پیکسل مجزا در ابتدا 0 میکنیم
                __m256 sum = _mm256_setzero_ps(); 
                int counter = 0; // شمارنده برای حرکت روی درایه‌های آرایه 9 عضوی کرنل
                // پیمایش درایه‌های کرنل 3x3 (پنجره اطراف پیکسل)
                for(int h = -1; h <= 1; h++) {
                    for(int w = -1; w <= 1; w++) {
                        // آدرس اولین کانال رنگی در پیکسل مبدا
                        int base = ((i + h) * width + (j + w)) * channels + c;
                        // خواندن وزن فعلی کرنل و کپی کردن آن در تمام 8 خانه رجیستر
                        //float kernelDer = kernel[counter++];
                        //معادل اسمبلی: vbroadcastss ymm, xmm
                        float kernelDer = kernel[counter++];
                        __m256 valuOfKernel = _mm256_set1_ps(kernelDer); 
                        
                        // با استفاده از آدری پایه و بردار پرش ها و طول پرش 4 بایتی ، بایت های هم رنگ را استخراج میکنیم برای هر پیکسل
                        // پارامتر آخر به منزله پرش های 4 بایتی به اندازه طول float است
                        __m256 vPixels = _mm256_i32gather_ps(&floatData[base], INDEX, 4);
                        // ضرب 8 پیکسل در وزن کرنل و جمع همزمان با مقادیر قبلی (FMA)
                        sum = _mm256_fmadd_ps(vPixels, valuOfKernel, sum);
                    }
                }
                
                // انتقال نتایج 8 پیکسل از رجیستر AVX2   به آرایه معمولی
                float results[8];
                _mm256_storeu_ps(results, sum);

                // ذخیره نتایج در تصویر خروجی
                for(int k = 0; k < 8; k++) {
                    float val = results[k];
                    
                    
                    // فیلترهای لبه مثل سوبل یا لاپلاسین اعداد منفی و بزرگ تولید می‌کنند
                    if(val < 0.0f) val = 0.0f;
                    if(val > 255.0f) val = 255.0f;// برای اینکه باید هر رنگ بین 0 تا 255 باشد و بیشتر نشود
                    
                    // محاسبه آدرس دقیق برای نوشتن در بافر خروجی
                    int out_idx = (i * inImage->width + (j + k)) * channels + c;
                    outImage->data[out_idx] = (unsigned char)val;
                }
            }
        }
    }

    // آزادسازی حافظه موقت
    free(floatData);
}

int main()
{
    // خواندن تصویر ورودی
    Image *image = readPicture("input.ppm");

    // تعریف کرنل تشخیص لبه 
    int kSize = 3;
    float kernel[9] = {
        -1, -1, -1, 
        -1,  8, -1, 
        -1, -1, -1
    };

    // تخصیص حافظه برای تصویر خروجی
    Image *outImage = (Image *)malloc(sizeof(Image));
    outImage->width = image->width;
    outImage->height = image->height;
    outImage->data = (unsigned char *)calloc(image->width * image->height * 3, 1);

    printf("Processing %dx%d color image...\n", image->width, image->height);

    // اجرای کانولوشن به روش استاندارد c
    printf("Running C Implementation...\n");

    clock_t start1 = clock();
    convolution2D(outImage, image, kernel, kSize);
    clock_t end1 = clock();
    double time_taken1 = ((double)(end1 - start1)) / CLOCKS_PER_SEC;
    printf("Time taken1: %f seconds\n", time_taken1);

    // اجرای کانولوشن با روش بهینه‌سازی شده avx2
    printf("Running Intrinsic SIMD Implementation (AVX2 Gather)...\n");
    clock_t start2 = clock();
    convolution2D_Intrinsic(outImage, image, kernel, kSize);
    clock_t end2 = clock();
    double time_taken2 = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time taken2: %f seconds\n", time_taken2);

    // ذخیره خروجی
    writePicture(outImage, "output.ppm");
    printf("Saved to 'output.ppm'\n");

    // آزادسازی حافظه
    free(image->data);
    free(image);
    free(outImage->data);
    free(outImage);
    
    return 0;
}