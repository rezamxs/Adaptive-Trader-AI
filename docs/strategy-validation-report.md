# گزارش اجرای آزمون‌ها و اعتبارسنجی فنی نرم‌افزار (Strategy Validation Report)

**پروژه:** Adaptive Trader AI  
**تاریخ گزارش:** ۲۱ مهر ۱۴۰۵ (۱۰ اکتبر ۲۰۲۶)  
**دستور اجرا:** `pytest backend/tests -v`  
**محیط اجرا:** Python 3.14.4، pytest 9.1.1، پلتفرم win32 (Windows 10 Pro 64-bit)  
**نتیجه کل آزمون‌ها:** **۵۵ آزمون PASSED (۱۰۰٪ قبولی)** — صفر آزمون FAILED یا SKIPPED  
**مدت‌زمان اجرا:** ۲۰.۲۷ ثانیه  

---

## ۱. جدول تفکیکی اجرای ۵۵ آزمون بر اساس ماژول‌ها

| ردیف | فایل آزمون | دسته‌بندی | تعداد آزمون | وضعیت | موارد آزموده‌شده |
| :---: | :--- | :---: | :---: | :---: | :--- |
| ۱ | `backend/tests/unit/test_scalper_strategy_rules.py` | قوانین اسکلپر | ۵ | **PASSED** | گیت حداقل سرمایه ۵,۰۰۰ دلاری طلا، فیلتر سشن زمانی، فیلتر پایان هفته، سقف زیان گرید، پارسر بومی DOCX |
| ۲ | `backend/tests/security/test_risk_and_circuit_breaker.py` | مدیریت ریسک و مدارشکن | ۵ | **PASSED** | پایداری حالت Emergency Stop در ری‌استارت، مسدودسازی سیگنال، فیلتر اسپرد و داده منقضی |
| ۳ | `backend/tests/security/test_demo_executor_security.py` | امنیت معاملات دمو | ۵ | **PASSED** | رد قطعی سفارشات لایو، الزام توکن تایید دمو، اعتبارسنجی پیش از ارسال |
| ۴ | `backend/tests/security/test_signal_mode_and_idempotency.py` | کنترل سیگنال | ۳ | **PASSED** | عملکرد مد `SIGNAL_ONLY`، کلید یکتایی (Idempotency) و عدم تکرار |
| ۵ | `backend/tests/data_ml/test_features_and_leakage.py` | داده و یادگیری ماشین | ۳ | **PASSED** | محاسبات شاخص‌های تکنیکال و عدم نشت داده‌های آینده (No Lookahead Leakage) |
| ۶ | `backend/tests/data_ml/test_ml_governance.py` | حاکمیت مدل‌ها | ۳ | **PASSED** | رجیستری مدل‌ها، ثبت متادیتا و جلوگیری از سوگیری آموزشی |
| ۷ | `backend/tests/integration/test_api_endpoints.py` | یکپارچگی وب‌سرویس | ۶ | **PASSED** | اندپوینت‌های مدیریت قوانین، اسناد، سیگنال‌ها، مدل‌ها و وضعیت سلامت سیستم |
| ۸ | `backend/tests/integration/test_db_models.py` | مدل‌های پایگاه داده | ۳ | **PASSED** | نگاشت جداول، ذخیره‌سازی، روابط کلیدهای خارجی و تراکنش‌ها |
| ۹ | `backend/tests/integration/test_mt5_provider.py` | ارائه‌دهنده MT5 | ۱ | **PASSED** | ارتباط با رابط متاتریدر ۵، نگاشت قیمت‌ها و ساختار کندل‌ها |
| ۱۰ | `backend/tests/unit/test_backtest_realistic.py` | موتور بک‌تست | ۴ | **PASSED** | شبیه‌سازی اسلیپیج، اسپرد و کمیسیون در بازدهی تاریخی |
| ۱۱ | `backend/tests/unit/test_core_config.py` | تنظیمات اصلی | ۵ | **PASSED** | اعتبارسنجی متغیرها و منع ساختاری تعریف `LIVE_TRADING` در Enum |
| ۱۲ | `backend/tests/unit/test_knowledge_pipeline.py` | خط پردازش دانش | ۴ | **PASSED** | استخراج قوانین از PDF و متن، ردیابی صفحات منبع و ابهامات |
| ۱۳ | `backend/tests/unit/test_market_data_mock.py` | داده‌های شبیه‌سازی | ۴ | **PASSED** | تولید مظنه‌ها و کندل‌های مصنوعی برای تست‌های ایزوله |
| ۱۴ | `backend/tests/unit/test_security_gates.py` | گیت‌های امنیتی | ۴ | **PASSED** | مسدودسازی مسیرهای معامله زنده و قفل‌های Fail-Closed |
| **کل** | **۱۴ فایل در ۴ پوشه** | — | **۵۵** | **۱۰۰٪ قبولی** | **تمامی اجزای سیستم با موفقیت آزموده شدند.** |

---

## ۲. خروجی ترمینال در زمان اجرای آزمون

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

======================= 55 passed, 1 warning in 20.27s ========================
```

---

## ۳. بررسی و رفع اشکال حین ممیزی

در جریان اجرای مجدد آزمون‌ها برای تایید صحت گزارش، آزمون `test_rules_api_list` با خطای اعتبارسنجی مواجه شد:  
- **علت خطا:** در ثبت اولیه ۶ قانون، مقادیر فیلد `category` به صورت حروف بزرگ (`RISK`، `FILTER`، `EXIT`) ذخیره شده بود، در حالی که اسکیماهای Pydantic در FastAPI مقادیر کوچک (`risk`، `filter`، `exit`) را الزامی می‌دانستند.  
- **اقدام اصلاحی:** مقادیر این فیلد در جدول `trading_rules` با اجرای کوئری اصلاحی به مقادیر استاندارد حروف کوچک تبدیل شدند.  
- **نتیجه:** اجرای مجدد آزمون با موفقیت کامل انجام شد و تمامی ۶ آزمون API با موفقیت پاس شدند.
