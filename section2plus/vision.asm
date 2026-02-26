global sobel_row_avx_asm
global convolve_window_avx_asm

section .data
    align 32
    two_vec:  times 8 dd 2.0        ; برداری شامل 8 عدد اعشاری 2.0 برای ضرب‌ها
    abs_mask: times 8 dd 0x7FFFFFFF ; ماسک بیتی برای محاسبه قدر مطلق (حذف بیت علامت)

section .text

; تابع اول: محاسبه یک سطر کامل از فیلتر سوبل با AVX2
; void sobel_row_avx_asm(const float* top, const float* mid, const float* bot, float* out, int width);
; ورودی‌ها در لینوکس (System V ABI):
; rdi = top_row (سطر بالایی)
; rsi = mid_row (سطر میانی)
; rdx = bot_row (سطر پایینی)
; rcx = out_row (خروجی)
; r8  = width (عرض تصویر)
sobel_row_avx_asm:
    ; آماده‌سازی ثبات‌های ثابت
    vmovaps ymm0, [two_vec]         ; ymm0 = [2.0, 2.0, ...]
    vmovaps ymm1, [abs_mask]        ; ymm1 = [0x7FFFFFFF, ...]

    mov r9, 1                       ; شمارنده x (از 1 شروع می‌شود تا حاشیه‌ها را رد کنیم)
    
.sobel_loop:
    ; شرط پایان: اگر x + 8 >= width شد، از حلقه خارج شو
    mov rax, r8                     ; rax = width
    sub rax, 8                      ; rax = width - 8
    cmp r9, rax                     
    jge .sobel_done

    ; محاسبه آفست حافظه: (x * 4 بایت)
    mov rax, r9
    shl rax, 2                      ; rax = x * 4

    ; --- محاسبه Gx (مشتق افقی) ---
    ; لود کردن پیکسل‌های سمت راست (x+1)
    vmovups ymm2, [rdi + rax + 4]   ; Top-Right
    vmovups ymm3, [rsi + rax + 4]   ; Mid-Right
    vmovups ymm4, [rdx + rax + 4]   ; Bot-Right

    ; لود کردن پیکسل‌های سمت چپ (x-1)
    vmovups ymm5, [rdi + rax - 4]   ; Top-Left
    vmovups ymm6, [rsi + rax - 4]   ; Mid-Left
    vmovups ymm7, [rdx + rax - 4]   ; Bot-Left

    ; محاسبات Gx
    vsubps ymm8, ymm2, ymm5         ; ymm8 = TR - TL
    vsubps ymm9, ymm3, ymm6         ; ymm9 = MR - ML
    vfmadd231ps ymm8, ymm0, ymm9    ; ymm8 = ymm8 + (2.0 * ymm9) [استفاده از FMA]
    vsubps ymm10, ymm4, ymm7        ; ymm10 = BR - BL
    vaddps ymm8, ymm8, ymm10        ; ymm8 = Gx نهایی

    ; --- محاسبه Gy (مشتق عمودی) ---
    ; لود کردن پیکسل‌های وسط (x)
    vmovups ymm11, [rdi + rax]      ; Top-Mid
    vmovups ymm12, [rdx + rax]      ; Bot-Mid

    ; محاسبات Gy
    vaddps ymm13, ymm5, ymm2        ; ymm13 = TL + TR
    vfmadd231ps ymm13, ymm0, ymm11  ; ymm13 += 2.0 * TM  (جمع سطر بالا)
    
    vaddps ymm14, ymm7, ymm4        ; ymm14 = BL + BR
    vfmadd231ps ymm14, ymm0, ymm12  ; ymm14 += 2.0 * BM  (جمع سطر پایین)

    vsubps ymm15, ymm14, ymm13      ; ymm15 = Bot_Sum - Top_Sum (Gy نهایی)

    ; --- قدر مطلق و جمع ---
    vandps ymm8, ymm8, ymm1         ; ymm8 = |Gx|
    vandps ymm15, ymm15, ymm1       ; ymm15 = |Gy|
    vaddps ymm8, ymm8, ymm15        ; ymm8 = |Gx| + |Gy|

    ; ذخیره در آرایه خروجی
    vmovups [rcx + rax], ymm8       ; edges[x] = ymm8

    ; پرش به 8 پیکسل بعدی
    add r9, 8
    jmp .sobel_loop

.sobel_done:
    ret

; تابع دوم: محاسبه کانولوشن (تطبیق الگو) برای یک پنجره مشخص
; float convolve_window_avx_asm(const float* img, const float* kernel, int img_width, int kSize);
; rdi = img (اشاره‌گر به گوشه بالا-چپ پنجره در تصویر)
; rsi = kernel (اشاره‌گر به ماتریس الگو)
; rdx = img_width (برای پرش به سطر بعدی تصویر)
; rcx = kSize (سایز الگو - حتماً باید مضربی از 8 باشد، مثلا 48)
; خروجی: نتیجه در ثبات xmm0 برگردانده می‌شود (استاندارد C برای Float)
convolve_window_avx_asm:
    vxorps ymm0, ymm0, ymm0         ; صفر کردن انباشتگر کلی (ymm0 = 0)
    
    mov r8, 0                       ; شمارنده سطرها (m)
.conv_row_loop:
    cmp r8, rcx
    jge .conv_done                  ; اگر m >= kSize، پایان

    mov r9, 0                       ; شمارنده ستون‌ها (n)
.conv_col_loop:
    cmp r9, rcx
    jge .conv_next_row              ; اگر n >= kSize، برو سطر بعد

    ; --- محاسبه آفست‌ها ---
    ; آفست تصویر = (m * img_width + n) * 4
    mov rax, r8
    imul rax, rdx                   ; rax = m * img_width
    add rax, r9                     ; rax = m * img_width + n
    shl rax, 2                      ; rax *= 4 (بایت)

    ; آفست کرنل = (m * kSize + n) * 4
    mov r10, r8
    imul r10, rcx                   ; r10 = m * kSize
    add r10, r9                     ; r10 = m * kSize + n
    shl r10, 2                      ; r10 *= 4 (بایت)

    ; --- لود و ضرب برداری (8 پیکسل همزمان) ---
    vmovups ymm1, [rdi + rax]       ; لود 8 پیکسل از تصویر
    vmovups ymm2, [rsi + r10]       ; لود 8 وزن از کرنل
    
    ; دستور FMA: ymm0 = ymm0 + (ymm1 * ymm2)
    vfmadd231ps ymm0, ymm1, ymm2

    add r9, 8                       ; پرش به 8 پیکسل بعدی
    jmp .conv_col_loop

.conv_next_row:
    inc r8                          ; m++
    jmp .conv_row_loop

.conv_done:
    ; در اینجا ymm0 شامل 8 عدد اعشاری است. باید آن‌ها را با هم جمع کنیم.
    ; استخراج مقادیر از YMM0 و جمع زدن به صورت افقی (Horizontal Add)
    vhaddps ymm0, ymm0, ymm0        ; جمع جفت‌های مجاور
    vhaddps ymm0, ymm0, ymm0        
    vextractf128 xmm1, ymm0, 1      ; جدا کردن نیمه بالایی رجیستر 256 بیتی
    addps xmm0, xmm1                ; جمع نیمه بالا و پایین
    
    ; مقدار نهایی اکنون در پایین‌ترین بخش xmm0 قرار دارد که استاندارد C است
    ret


# 1. کامپایل فایل اسمبلی
;nasm -f elf64 vision.asm -o vision.o

# 2. کامپایل فایل C
;gcc -c main.c -o main.o -mavx2 -O3

# 3. پیوند دادن (Linking) هر دو با هم و ساخت فایل اجرایی
;gcc main.o vision.o -o project -lm

# 4. اجرا
;./project
