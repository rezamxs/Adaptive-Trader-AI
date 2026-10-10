# ممیزی ریاضی و اعتبارسنجی الگوریتمی Volume Profile
## سامانه Adaptive Trader AI

این سند تحلیل موشکافانه فرمول‌ها، الگوریتم‌های گسسته‌سازی، قضایای ریاضی بقای حجم و رفتار حالات مرزی در کلاس [VolumeProfileCalculator](file:///g:/mash/backend/app/services/dataset/volume_profile.py) را ارائه می‌دهد.

---

### ۱. چارچوب گسسته‌سازی قیمت (Discretization & Binning)

داده‌های ورودی شامل مجموعه‌ای از کندل‌های قیمتی با مشخصات $[\text{Low}_t, \text{High}_t, \text{Volume}_t]$ هستند. برای ساخت توزیع فرکانس انباشتی حجم، محور پیوسته قیمت به سطل‌های گسسته (Bins) تقسیم می‌شود:

$$\text{min\_bin} = \lfloor \frac{\min(\text{Low})}{\text{price\_step}} \rfloor \times \text{price\_step}$$
$$\text{max\_bin} = \lceil \frac{\max(\text{High})}{\text{price\_step}} \rfloor \times \text{price\_step}$$
$$N_{\text{bins}} = \text{round}\left(\frac{\text{max\_bin} - \text{min\_bin}}{\text{price\_step}}\right) + 1$$

مراکز سطل‌ها به صورت بردار زیر شکل می‌گیرند:
$$\mathbf{P} = \{ p_0, p_1, \dots, p_{N-1} \} \quad \text{where} \quad p_k = \text{min\_bin} + k \times \text{price\_step}$$

انتخاب `price_step` وابسته به ابعاد مقیاس قیمت نماد است:
- برای طلا (`XAUUSD`): با اسپرد معمولی ۳۰ تا ۴۰ سنت، مقدار بهینه `price_step = 0.10` یا `0.20` است.
- برای یورو به دلار (`EURUSD`): با اسپرد حدود ۰.۷ پیپ، مقدار `price_step = 0.0001` یا `0.0002` در نظر گرفته می‌شود.

---

### ۲. مدل تخصیص حجم: تخمین توزیع یکنواخت بازه‌ای (Uniform Range Estimate)

در کتابخانه‌های ساده‌انگارانه تحلیل تکنیکال، کل حجم کندل صرفاً به قیمت بسته شدن ($\text{Close}$) تخصیص می‌یابد. این فرض از نظر ساختار بازار دچار سوگیری شدید است، زیرا قیمت در طول طول عمر کندل بین کمترین و بیشترین مقدار نوسان کرده است.

در پیاده‌سازی [VolumeProfileCalculator](file:///g:/mash/backend/app/services/dataset/volume_profile.py)، حجم کندل $v_t$ به صورت یکنواخت بین سطل‌هایی که بازه $[\text{Low}_t, \text{High}_t]$ را پوشش می‌دهند توزیع می‌شود:

$$\text{start\_idx}_t = \max\left(0, \lfloor \frac{\text{Low}_t - \text{min\_bin}}{\text{price\_step}} \rfloor\right)$$
$$\text{end\_idx}_t = \min\left(N-1, \lceil \frac{\text{High}_t - \text{min\_bin}}{\text{price\_step}} \rceil\right)$$
$$\text{span\_count}_t = \max(1, \text{end\_idx}_t - \text{start\_idx}_t + 1)$$

تخصیص حجم به هر سطل $k \in [\text{start\_idx}_t, \text{end\_idx}_t]$:
$$\Delta V_k = \frac{v_t}{\text{span\_count}_t}$$

#### قضیه بقای حجم (Volume Conservation)
مجموع حجم‌های تخصیص‌یافته به تمام سطل‌ها دقیقاً برابر با حجم ورودی کندل است:
$$\sum_{k=\text{start\_idx}_t}^{\text{end\_idx}_t} \Delta V_k = \sum_{k=\text{start\_idx}_t}^{\text{end\_idx}_t} \frac{v_t}{\text{span\_count}_t} = \text{span\_count}_t \times \frac{v_t}{\text{span\_count}_t} = v_t$$
بنابراین:
$$\sum_{k=0}^{N-1} V_k = \sum_{t=1}^{T} v_t$$
این قضیه در آزمون واحد `test_volume_profile_volume_conservation` به طور عددی اثبات و اعتبارسنجی شده است.

---

### ۳. الگوریتم محاسبه نقطه کنترل (POC) و حل تساوی (Tie-Breaking)

نقطه کنترل، سطحی است که بیشترین حجم انباشته را داراست:
$$k_{\text{POC}} = \arg\max_{k} V_k$$
$$\text{POC} = p_{k_{\text{POC}}}$$

**قانون رفتار در تساوی:**
اگر چند سطل قیمتی دارای حداکثر حجم یکسان باشند، تابع `np.argmax` نخستین ایندکسی را که به این مقدار می‌رسد برمی‌گزیند. از آنجا که ایندکس‌ها از پایین‌ترین قیمت به بالاترین قیمت چیده شده‌اند، در شرایط تساوی همواره پایین‌ترین سطح قیمت به عنوان POC انتخاب می‌شود. این رفتار قطعی (Deterministic) بوده و در آزمون `test_volume_profile_tie_breaking_deterministic` بررسی و ثبت شده است.

---

### ۴. الگوریتم دوطرفه محاسبه محدوده ارزش (Value Area: VAH & VAL)

محدوده ارزش معرف بازه‌ای است که ۷۰ درصد از حجم کل معاملات را در بر می‌گیرد ($V_{\text{target}} = 0.70 \times V_{\text{total}}$):
1. الگوریتم از سطل $k_{\text{POC}}$ آغاز می‌شود و دو شاخص $\text{low\_i} = k_{\text{POC}}$ و $\text{high\_i} = k_{\text{POC}}$ تنظیم می‌گردند.
2. در هر گام، حجم سطل یک پله بالاتر ($V_{\text{high\_i} + 1}$) با حجم سطل یک پله پایین‌تر ($V_{\text{low\_i} - 1}$) مقایسه می‌شود.
3. سطحی که حجم بیشتری دارد جذب محدوده ارزش شده و شاخص مربوطه یک واحد گسترش می‌یابد:
   - اگر $V_{\text{high\_i} + 1} \ge V_{\text{low\_i} - 1}$ باشد، شاخص بالایی رشد می‌کند (`high_i += 1`).
   - در غیر این صورت، شاخص پایینی گسترش می‌یابد (`low_i -= 1`).
4. حلقه تا زمانی که حجم تجمعی به $V_{\text{target}}$ برسد ادامه می‌یابد.
5. در نهایت:
   $$\text{VAL} = p_{\text{low\_i}}, \quad \text{VAH} = p_{\text{high\_i}}$$

رعایت شرط صوری و ریاضی $\text{VAL} \le \text{POC} \le \text{VAH}$ همواره تضمین‌شده است.

---

### ۵. تحلیل شرایط مرزی و لبه‌ای (Edge Cases)

| حالت مرزی | رفتار الگوریتم | پیامد و اعتبار |
| :--- | :--- | :--- |
| **کندل تخت ($High == Low$)** | شاخص سطل مستقیماً با رند کردن محاسبه و کل حجم به همان سطل واریز می‌شود. | بدون خطای تقسیم بر صفر؛ $\text{VAL} = \text{POC} = \text{VAH}$ |
| **کل حجم صفر ($V_{\text{total}} = 0$)** | خروجی مقادیر `poc=None`, `val=None`, `vah=None` است. | جلوگیری از محاسبات بی‌معنا و بدون کرش نرم‌افزاری |
| **دامنه بسیار کوچک‌تر از گام** | سطل‌های کف و سقف هم‌پوشان می‌شوند و حجم پایسته می‌ماند. | رفتار پایدار و بدون پرش غیرعادی |
| **دیتافریم خالی** | دیکشنری با مقادیر تهی بازگردانده می‌شود. | مدیریت بدون خطا در خط لوله |
