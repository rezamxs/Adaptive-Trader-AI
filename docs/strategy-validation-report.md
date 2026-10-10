# گزارش آزمون‌های خودکار و اعتبارسنجی نرم‌افزاری (Strategy Validation Report)

**پروژه:** Adaptive Trader AI  
**تاریخ گزارش:** اکتبر ۲۰۲۶  
**محیط آزمون:** پایتون ۳.۱۴، pytest ۹.۱.۱، هر دو محیط Windows x64 و پایگاه داده ایزوله حافظه‌ای SQLite  
**نرخ موفقیت:** **۱۰۰٪ (۵۵ آزمون موفق از ۵۵ آزمون)** — صفر آزمون ناموفق  

---

## ۱. جدول خلاصه نتایج اجرای آزمون‌های خودکار (`pytest backend/tests -v`)

| رده آزمون | فایل آزمون | تعداد کل | نتیجه | پوشش و هدف اعتبارسنجی |
| :--- | :--- | :---: | :---: | :--- |
| **قوانین اسکلپر** | `backend/tests/unit/test_scalper_strategy_rules.py` | ۵ | **PASS** | اعتبارسنجی گیت حداقل سرمایه ۵,۰۰۰ دلاری طلا، فیلتر سشن آسیا/اروپا، فیلتر گپ جمعه، سقف دراودان ۲۰۰ دلاری گرید، و پارسر بومی DOCX. |
| **امنیت و مدارشکن** | `backend/tests/security/test_risk_and_circuit_breaker.py` | ۵ | **PASS** | پایداری وضعیت توقف اضطراری (Circuit Breaker)، مسدودسازی سیگنال در حالت اضطراری، فیلتر اسپرد و داده‌های منقضی. |
| **ایمن‌سازی دمو** | `backend/tests/security/test_demo_executor_security.py` | ۵ | **PASS** | رد قطعی حساب لایو، تضمین توکن `ALLOW_DEMO_ORDER_EXECUTION`، چک‌های پیش از معامله. |
| **سیگنال و تطابق** | `backend/tests/security/test_signal_mode_and_idempotency.py` | ۳ | **PASS** | تولید سیگنال در حالت `SIGNAL_ONLY`، تضمین یکتایی و کلید عدم ارسال تکراری (Idempotency). |
| **داده و یادگیری ماشین** | `backend/tests/data_ml/test_features_and_leakage.py` | ۳ | **PASS** | محاسبات شاخص‌های تکنیکال و تضمین عدم نشت داده از آینده (No Data Leakage). |
| **حاکمیت مدل‌ها** | `backend/tests/data_ml/test_ml_governance.py` | ۳ | **PASS** | رجیستری مدل‌ها، اعتبارسنجی متامدل و جلوگیری از سوگیری در آموزش. |
| **یکپارچگی API** | `backend/tests/integration/test_api_endpoints.py` | ۶ | **PASS** | عملکرد اندپوینت‌های مدیریت اسناد، قوانین، سیگنال‌ها، مدل‌ها و سامانه. |
| **مدل‌های پایگاه داده** | `backend/tests/integration/test_db_models.py` | ۳ | **PASS** | ذخیره‌سازی، روابط، نگاشت جداول و مهاجرت داده‌ها. |
| **یکپارچگی MT5** | `backend/tests/integration/test_mt5_provider.py` | ۱ | **PASS** | شبیه‌سازی اتصال کارگزار، نگاشت داده‌ها و دریافت مظنه‌ها. |
| **بک‌تست واقع‌گرایانه** | `backend/tests/unit/test_backtest_realistic.py` | ۴ | **PASS** | شبیه‌سازی دقیق اسلیپیج، اسپرد و کمیسیون در موتور بک‌تست. |
| **تنظیمات پایه** | `backend/tests/unit/test_core_config.py` | ۵ | **PASS** | تنظیمات محیطی، ساختار دایرکتوری‌ها و متغیرهای پیکربندی امن. |
| **خط دانش و اسناد** | `backend/tests/unit/test_knowledge_pipeline.py` | ۴ | **PASS** | پارس اسناد PDF و متنی، استخراج قوانین و ردیابی ابهامات. |
| **داده‌های مصنوعی** | `backend/tests/unit/test_market_data_mock.py` | ۴ | **PASS** | تولید داده‌های واقع‌گرایانه قیمت جهت تست‌های بدون شبکه. |
| **گیت‌های امنیتی** | `backend/tests/unit/test_security_gates.py` | ۴ | **PASS** | مسدودسازی مسیرهای معامله زنده و قفل‌های ساختاری نرم‌افزار. |
| **مجموع کل** | **۱۴ فایل تست در ۴ دایرکتوری** | **۵۵** | **۱۰۰٪ قبولی** | **سیستم کاملاً پایدار، آزمون‌شده و آماده است.** |

---

## ۲. خروجی رسمی دستور اجرای آزمون‌ها

```text
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
rootdir: G:\mash
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 55 items

backend\tests\data_ml\test_features_and_leakage.py ...                   [  5%]
backend\tests\data_ml\test_ml_governance.py ...                          [ 10%]
backend\tests\integration\test_api_endpoints.py ......                   [ 21%]
backend\tests\integration\test_db_models.py ...                          [ 27%]
backend\tests\integration\test_mt5_provider.py .                         [ 29%]
backend\tests\security\test_demo_executor_security.py .....              [ 38%]
backend\tests\security\test_risk_and_circuit_breaker.py .....            [ 47%]
backend\tests\security\test_signal_mode_and_idempotency.py ...           [ 52%]
backend\tests\unit\test_backtest_realistic.py ....                       [ 60%]
backend\tests\unit\test_core_config.py .....                             [ 69%]
backend\tests\unit\test_knowledge_pipeline.py ....                       [ 76%]
backend\tests\unit\test_market_data_mock.py ....                         [ 83%]
backend\tests\unit\test_scalper_strategy_rules.py .....                  [ 92%]
backend\tests\unit\test_security_gates.py ....                           [100%]

======================= 55 passed, 1 warning in 25.01s ========================
```

---

## ۳. تحلیل رفتار گیت‌های امنیتی پیاده‌سازی‌شده

۱. **آزمون گیت حداقل سرمایه طلا (`test_gold_minimum_equity_gate`):**
   - برای بالانس ۴,۹۰۰ دلار، درخواست با خطای قطعی `REJECTED_GOLD_MIN_EQUITY` مسدود شد و حجم اختصاص‌یافته برابر ۰.۰ لات بود.
   - برای بالانس ۱۰,۰۰۰ دلار، درخواست با موفقیت به حجم مجاز تبدیل و تایید شد.
۲. **آزمون فیلتر زمانی سشن (`test_session_hours_gate`):**
   - در ساعت ۰۵:۰۰ UTC (سشن آسیا/اروپا)، سیگنال با موفقیت تایید شد.
   - در ساعت ۱۴:۰۰ UTC (سشن نیویورک)، سیگنال با خطای `REJECTED_SESSION_OUTSIDE_ALLOWED_HOURS` مسدود شد.
۳. **آزمون گپ پایان هفته (`test_friday_close_weekend_risk`):**
   - در روز جمعه ساعت ۱۹:۳۰ UTC، سیگنال به دلیل قرار داشتن در ساعات پرریسک پایان هفته رد شد (`REJECTED_FRIDAY_WEEKEND_RISK`).
۴. **آزمون سقف دراودان گرید (`test_martingale_drawdown_ceiling`):**
   - با دراودان ۱۵۰ دلار، سیگنال در محدوده مجاز ماند.
   - با دراودان ۲۰۰ دلار، سیگنال بی‌درنگ با خطای `REJECTED_MARTINGALE_LOSS_LIMIT` رد شد.
۵. **آزمون پارسر بومی اسناد Word (`test_document_parser_docx`):**
   - فایل واقعی `اصطلاح پرکاربرد در ترید.docx` بدون هیچ کتابخانه جانبی باز شد و ۳۱۳ پاراگراف با موفقیت استخراج و بازبینی شدند.
