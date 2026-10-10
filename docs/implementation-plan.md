# برنامه اجرایی تفصیلی سامانه Adaptive Trader AI (Implementation Plan)
**نسخه:** 1.0.0  
**رویکرد مهندسی:** توسعه ماژولار گام‌به‌گام با آزمون خودکار در انتهای هر مرحله

---

## ماتریس مراحل پیاده‌سازی و معیارهای پذیرش

| فاز | عنوان فاز | ماژول‌ها و فایل‌های کلیدی | معیار پذیرش (Acceptance Criteria) |
| :--- | :--- | :--- | :--- |
| **Phase A** | بازرسی محیط سیستم | `docs/environment-audit.md` | ثبت اطلاعات دقیق سخت‌افزار، Python، MT5، اثبات دمو بودن با `trade_mode=0` |
| **Phase B** | تدوین معماری و برنامه | `docs/architecture.md`, `docs/implementation-plan.md` | تایید ساختار ماژولار، سناریوهای Fail-Closed و مسیرهای آزمون |
| **Phase C** | اسکلت هسته و پایگاه داده | `backend/app/core/`, `backend/app/db/`, `backend/app/models/` | ایجاد دیتابیس با مدل‌های SQLAlchemy، مایگریشن اولیه، ثبت لاگ و آزمون ارتباط |
| **Phase D** | دریافت دانش و قوانین | `backend/app/services/knowledge/`, تست‌های پارسر | پردازش PDF/TXT/Excel، استخراج قوانین ساختاریافته، حفظ منبع و گردشکار تایید |
| **Phase E** | ارائه‌دهنده داده بازار MT5 | `backend/app/services/market_data/` | واسط Read-Only، دریافت کندل تاریخی، مدیریت خطا و تشخیص داده کهنه/ناقص |
| **Phase F** | مهندسی ویژگی و دیتاست | `backend/app/services/dataset/` | محاسبه شاخص‌ها بدون نشت آینده، برچسب‌گذاری محتاطانه TP/SL، تفکیک زمانی |
| **Phase G** | مدل یادگیری ماشین و نسخهبندی | `backend/app/services/ml/` | آموزش مدل پایه، ارزیابی Walk-Forward، کالیبراسیون و ثبت در ریجستری |
| **Phase H** | موتور بکتست واقع‌گرایانه | `backend/app/services/backtest/` | شبیه‌سازی با اسپرد، کمیسیون، اسلیپیج و گزارش شاخص‌های MaxDD و Profit Factor |
| **Phase I** | موتور ریسک و کلید توقف | `backend/app/services/risk/` | محاسبه حجم، سقف دراودان روزانه، قفل اضطراری پایدار و مسدودسازی امن |
| **Phase J** | تولید سیگنال (`SIGNAL_ONLY`) | `backend/app/services/signal/` | تولید سیگنال با کلید Idempotency، عدم ارسال سفارش و ذخیره کامل وضعیت |
| **Phase K** | ماژول معاملات دمو | `backend/app/services/execution/` | کنترل پنج‌گانه امنیتی، رد حساب غیر دمو، رد در صورت کلید توقف یا قطع داده |
| **Phase L** | داشبورد فارسی و API | `backend/app/api/`, `frontend/` | داشبورد راست‌چین (RTL)، نمایش کارت‌های وضعیت زنده، توقف اضطراری و مدیریت قوانین |
| **Phase M** | آزمون جامع و مستندسازی | `backend/tests/`, `docs/testing.md`, `docs/limitations.md` | اجرای ۴ دسته آزمون (Unit, Integration, Security, Data)، ثبت مستندات نهایی |

---

## دستورات آزمون خودکار در هر مرحله

- اجرای آزمون‌های واحد: `pytest backend/tests/unit -v`
- اجرای آزمون‌های یکپارچه‌سازی: `pytest backend/tests/integration -v`
- اجرای آزمون‌های امنیتی و معاملاتی: `pytest backend/tests/security -v`
- اجرای آزمون‌های داده و نشت اطلاعات: `pytest backend/tests/data_ml -v`
- اجرای کل سوئیت آزمون: `pytest backend/tests -v`
