# خط پایه معماری و خط لوله داده Volume Profile
## سامانه Adaptive Trader AI

این سند ساختار معماری، جریان داده و خط پایه ماژول بومی Volume Profile را در سامانه معین می‌کند.

---

### ۱. جایگاه ماژول در معماری سامانه

ماژول [volume_profile.py](file:///g:/mash/backend/app/services/dataset/volume_profile.py) به عنوان یک مؤلفه مهندسی ویژگی و تحلیل حراج قیمت درون بسته `dataset` قرار دارد. هدف این مؤلفه محاسبه سطوح کلیدی نظریه حراج بازار شامل:
- **نقطه کنترل (Point of Control - POC):** سطح قیمتی دارای بیشترین تراکم حجم مبادلاتی.
- **محدوده ارزش بالا (Value Area High - VAH):** کرانه بالایی بازه قیمتی دربرگیرنده ۷۰ درصد از کل حجم دادوستدها.
- **محدوده ارزش پایین (Value Area Low - VAL):** کرانه پایینی بازه قیمتی دربرگیرنده ۷۰ درصد از کل حجم دادوستدها.

```
+------------------------------------+
|  MetaTrader 5 (MT5MarketDataProvider) |
+------------------------------------+
                 | (copy_rates_from_pos, start_pos=1)
                 v
+------------------------------------+
|         DataFrame کندل‌های بسته       |
+------------------------------------+
                 |
                 +-----------------------------------+
                 |                                   |
                 v                                   v
+---------------------------------+  +---------------------------------+
|   FeatureEngineer (ویژگی‌های پایه) |  | VolumeProfileCalculator (نامزد) |
| RSI, ATR, Geometry, Return, EMA |  | POC, VAH, VAL, Value Area Dist  |
+---------------------------------+  +---------------------------------+
                 |                                   |
                 +-----------------+-----------------+
                                   |
                                   v
                 +-----------------------------------+
                 |          DatasetPipeline          |
                 |     (Purging / Embargo / Scaler)  |
                 +-----------------------------------+
                                   |
                                   v
                 +-----------------------------------+
                 |    ModelTrainer (مدل‌های کاندید)    |
                 |      LogisticRegression / RF      |
                 +-----------------------------------+
```

---

### ۲. نحوه جریان داده از تامین‌کننده MT5

تامین داده در سامانه از طریق کلاس [MT5MarketDataProvider](file:///g:/mash/backend/app/services/market_data/mt5_provider.py) انجام می‌پذیرد.
در این کلاس، تابع `get_historical_dataframe` فراخوانی می‌شود که به طور پیش‌فرض از پارامتر `start_pos=1` در متد `copy_rates_from_pos` استفاده می‌کند:

```python
rates = mt5.copy_rates_from_pos(symbol, mt5_tf, start_pos=1, count=count)
```

**سیاست حیاتی کندل ناقص (Incomplete Current Bar):**
- کندل ایندکس ۰ در MetaTrader 5 کندلی است که هنوز بسته نشده و قیمت‌های High، Low، Close و ستون `tick_volume` آن لحظه‌به‌لحظه تغییر می‌کنند.
- وارد کردن کندل جاری در محاسبات تاریخی منجر به تغییر متغیرها در طول زمان و نشت اطلاعات (Data Leakage) می‌شود.
- با تنظیم `start_pos=1`، تنها کندل‌های قطعی و بسته‌شده (`is_closed=True`) وارد آرایه پردازش می‌شوند.

---

### ۳. ویژگی‌های خروجی و نحوه نرمال‌سازی

کلاس [VolumeProfileCalculator](file:///g:/mash/backend/app/services/dataset/volume_profile.py) از متد `compute_rolling_profile_features` برای تولید متغیرهای غلتان بدون نگاه به آینده (Zero Lookahead) بهره می‌برد.
برای هر کندل $i$ در پنجره زمانی گذشته ($[i - \text{window} : i]$):

1. **`vp_poc`**: سطح قیمت نقطه کنترل در پنجره گذشته.
2. **`vp_vah`**: سطح کرانه بالای محدوده ارزش.
3. **`vp_val`**: سطح کرانه پایین محدوده ارزش.
4. **`dist_vp_poc`**: فاصله نسبی قیمت بسته با POC به صورت:
   $$\text{dist\_vp\_poc} = \frac{\text{Close}_i - \text{POC}}{\text{Close}_i}$$
5. **`dist_vp_vah`**: فاصله نسبی قیمت بسته با VAH به صورت:
   $$\text{dist\_vp\_vah} = \frac{\text{Close}_i - \text{VAH}}{\text{Close}_i}$$
6. **`dist_vp_val`**: فاصله نسبی قیمت بسته با VAL به صورت:
   $$\text{dist\_vp\_val} = \frac{\text{Close}_i - \text{VAL}}{\text{Close}_i}$$
7. **`in_value_area`**: متغیر دوحالته (باینری) که نشان می‌دهد آیا قیمت بسته درون محدوده ارزش قرار دارد یا خیر:
   $$\mathbb{I}(\text{VAL} \le \text{Close}_i \le \text{VAH})$$

---

### ۴. مرزهای امنیتی و وضعیت سیگنال

مطابق الزامات ایمنی سامانه:
- ماژول به صورت ایزوله طراحی شده و هیچ تغییری در موتور هسته تصمیم‌گیری زنده ایجاد نمی‌کند.
- متغیرهای تولیدی صرفاً به عنوان متغیرهای پژوهشی (Research Features) ارزیابی می‌شوند.
- اجرای سامانه کماکان در وضعیت `SIGNAL_ONLY` باقی مانده و ارسال هرگونه سفارش معاملاتی غیرفعال است.
