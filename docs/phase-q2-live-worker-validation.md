# گزارش اعتبارسنجی فاز Q2: صحت پردازش کندل، پایداری Worker و تفکیک وضعیت بازار
## سامانه Adaptive Trader AI | مسیر `G:\mash`

---

## ۱. خلاصه اجرایی

در فاز Q2، رفتار سرویس پس‌زمینه پایش بازار (`MarketMonitoringWorker`) از ابعاد دریافت داده جدید، پیشگیری قطعی از پردازش تکراری (Deduplication)، اتمیک بودن چرخه حیات کندل، تفکیک وضعیت اتصال از تازگی مظنه، ایمنی چندنخی در فراخوانی‌های C-extension متاتریدر ۵ و پایداری در زمان راه‌اندازی مجدد (Restart) به صورت جامع ارزیابی و تقویت گردید.

بررسی داده‌های زنده ترمینال MetaTrader 5 نشان داد که اتصال به ترمینال برقرار است (`MT5 Connected = True`، لاگین حساب دمو ۵۸۰۳۳۵۸۳ در کارگزار `YourBroker-MT5`)، اما به دلیل زمان اجرای آزمون در پایان هفته (یکشنبه، ۱۱ اکتبر ۲۰۲۶)، آخرین تیک ثبت‌شده در سرور متعلق به جمعه ۹ اکتبر ۲۰۲۶ ساعت ۲۳:۵۴:۵۹ UTC با کهنگی حدود ۳۹.۸ ساعت است. بازار طلا (`XAUUSD`) در تعطیلات آخر هفته قرار دارد و هیچ کندل بسته جدیدی در بازار زنده تولید نمی‌شود.

بر اساس دستورالعمل فاز، اعتبارسنجی به دو بخش مستقل تفکیک شد:
۱. **اعتبارسنجی نرم‌افزاری و یکپارچگی (Mock & Integration Validation):** با موفقیت ۱۰۰٪ کامل (۱۲ آزمون اختصاصی جدید در [backend/tests/unit/test_phase_q2_worker_validation.py](file:///g:/mash/backend/tests/unit/test_phase_q2_worker_validation.py) و در مجموع ۹۷ آزمون کل پروژه بدون شکست پاس شدند).
۲. **اعتبارسنجی زنده بازار (Live Market Validation):** وضعیت ترمینال تأیید شد، اما دریافت کندل‌های بسته زنده به بازگشایی نشست معاملاتی موکول گردید و وضعیت کلی این بخش «اعتبارسنجی زنده نیازمند بازگشایی نشست بازار» ثبت شد.

تعداد کل سفارش‌های معاملاتی ارسالی به بازار در تمام مراحل این فاز **اکیداً صفر** بوده است.

---

## ۲. وضعیت اولیه و تغییرات انجام‌شده

### وضعیت اولیه در شروع فاز Q2
- Worker در فاز Q1 پیاده‌سازی شده بود اما ثبت چرخه در پایگاه داده تنها در انتهای محاسبات سنگین رخ می‌داد؛ در نتیجه ریسک رقابت (Race Condition) در پردازش‌های همزمان برای یک کندل وجود داشت.
- متاتریدر ۵ بدون هماهنگی نخی (Thread Lock) در استخر نخ‌های پیش‌فرض پایتون فراخوانی می‌شد.
- وضعیت‌های بازار به صورت خطی بررسی می‌شدند و تعطیلی بازار صرفاً از روی کهنگی تیک سنجیده می‌شد، بدون تفکیک دقیق وضعیت اتصال از وضعیت نشست معاملاتی.
- جدول پایگاه داده `worker_cycle_records` فاقد ستون ردیابی فاز چرخه عمر (`lifecycle_phase`) بود.

### تغییرات اعمال‌شده در فاز Q2
1. **ارتقای مدل داده و پایگاه داده:** ستون‌های `lifecycle_phase` و `updated_at` به جدول [backend/app/models/worker_cycle.py](file:///g:/mash/backend/app/models/worker_cycle.py) اضافه شدند و سازوکار ارتقای خودکار ساختار در [backend/app/db/init_db.py](file:///g:/mash/backend/app/db/init_db.py) پیاده‌سازی شد.
2. **قفل‌گذاری چندنخی MT5:** قفل سراسری بازگشتی `_mt5_lock = threading.RLock()` در [backend/app/services/market_data/mt5_provider.py](file:///g:/mash/backend/app/services/market_data/mt5_provider.py) اضافه شد تا تمام فراخوانی‌های C-extension متاتریدر ۵ سریالی و کاملاً ایمن شوند.
3. **رزرو اتمیک دومرحله‌ای کندل:** چرخه حیات `DISCOVERED -> PROCESSING -> COMPLETED / FAILED` در [backend/app/services/worker/market_worker.py](file:///g:/mash/backend/app/services/worker/market_worker.py) اضافه شد. پیش از ورود به تحلیل سنگین، کندل با قید یکتایی `bar_key` در دیتابیس رزرو می‌شود تا حتی در صورت فراخوانی همزمان، اجرای محاسبات تکراری غیرممکن گردد.
4. **دسته‌بندی جامع وضعیت بازار:** متد مستقل `evaluate_market_freshness` برای تفکیک حالات `MT5_CONNECTED`، `MT5_DISCONNECTED`، `MARKET_OPEN_DATA_FRESH`، `NO_RECENT_TICK`، `MARKET_CLOSED`، `MARKET_CLOSED_OR_STALE` و `DATA_INVALID` طراحی شد.
5. **ارتقای اندپوینت‌های API:** اطلاعات فاز چرخه عمر و تفکیک وضعیت اتصال و بازار در [backend/app/api/v1/endpoints/system.py](file:///g:/mash/backend/app/api/v1/endpoints/system.py) منعکس گردید.
6. **مجموعه آزمون‌های تخصصی:** طراحی و پیاده‌سازی ۱۲ سناریوی آزمون جامع در [backend/tests/unit/test_phase_q2_worker_validation.py](file:///g:/mash/backend/tests/unit/test_phase_q2_worker_validation.py).

---

## ۳. تنظیمات امنیتی قبل و بعد از آزمون

تنظیمات امنیتی پیش و پس از اجرای آزمون‌ها مورد بازرسی دقیق قرار گرفت و هیچ تغییری در آنها ایجاد نشد:

| پارامتر امنیتی | مقدار قبل از آزمون | مقدار بعد از آزمون | وضعیت انطباق |
| :--- | :--- | :--- | :--- |
| `EXECUTION_MODE` | `SIGNAL_ONLY` | `SIGNAL_ONLY` | تأیید شده (Fail-Closed) |
| `DEMO_TRADING_ENABLED` | `False` | `False` | تأیید شده (غیرفعال قطعی) |
| `WORKER_ENABLED` | `True` | `True` | فعال در بستر پایش |
| `MT5 Trade Allowed` | `False` | `False` | بدون تغییر در ترمینال |
| قوانین معاملاتی تأییدشده | ۰ قانون | ۰ قانون | هیچ قانون pending تأیید نشد |
| مدل‌های یادگیری ماشین فعال | ۰ مدل | ۰ مدل | هیچ مدل ردشده‌ای فعال نشد |
| تعداد سفارش‌های ارسالی | ۰ | ۰ | صفر سفارش (اثبات‌شده) |

---

## ۴. معماری Worker و نحوه ثبت چرخه‌ها

### چرخه عمر چندمرحله‌ای (Lifecycle Phases)
چرخه پردازش هر کندل به شرح زیر اتمیک شده است:

```
[دریافت کندل بسته start_pos=1]
               │
               ▼
   [محاسبه bar_key یکتا]
               │
               ├───────────────────────────────────────────────┐
               ▼                                               ▼
       [بررسی در دیتابیس]                              [یافت شد: COMPLETED]
               │                                               │
               ├───────────────┐                               ▼
               ▼               ▼                       [NO_NEW_CLOSED_BAR]
         [یافت نشد]      [یافت شد: PROCESSING]           (بدون تحلیل مجدد)
               │               │
               │               ▼
               │       [کمتر از ۱۲۰ ثانیه: رد به عنوان در حال پردازش]
               │       [بیش از ۱۲۰ ثانیه: بازیابی رکورد رهاشده]
               │
               ▼
[ثبت اتمیک رکورد با وضعیت PROCESSING]
  (در صورت خطای همزمانی IntegrityError: عقب‌نشینی به NO_NEW_CLOSED_BAR)
               │
               ▼
 [محاسبه ویژگی‌ها، مدل و قوانین]
               │
               ├───────────────────────────────┐
               ▼                               ▼
       [موفقیت‌آمیز]                         [خطای تحلیل]
               │                               │
               ▼                               ▼
 [به‌روزرسانی رکورد: COMPLETED]     [به‌روزرسانی رکورد: FAILED]
```

### تفاوت تضمین «جلوگیری از اجرای دوباره تحلیل» و «جلوگیری از ثبت دوباره نتیجه»
- **جلوگیری از ثبت دوباره نتیجه:** با شاخص یکتای `unique=True` روی ستون `bar_key` در جدول SQLite تضمین شده است و به هیچ وجه امکان ثبت دو نتیجه برای یک کندل وجود ندارد.
- **جلوگیری از اجرای دوباره تحلیل:** با سازوکار جدید «رزرو اولیه در مرحله PROCESSING»، پیش از شروع محاسبات ریاضی ویژگی‌ها، رکورد اولیه در دیتابیس Commit می‌شود. بنابراین هر پردازش دیگر یا حلقه بعدی در همان لحظه متوجه رزرو بودن کندل شده و محاسبات سنگین را تکرار نمی‌کند. تنها حالتی که تحلیل ممکن است تکرار شود، زمانی است که سیستم دقیقاً در اواسط تحلیل کرش کند و پس از دو دقیقه منقضی شدن رزرو، پردازش بازیابی گردد.

---

## ۵. نتایج آزمون‌های مستقل Mock

آزمون‌های شبیه‌سازی‌شده در بستر ایزوله [backend/tests/unit/test_phase_q2_worker_validation.py](file:///g:/mash/backend/tests/unit/test_phase_q2_worker_validation.py) با موتور دیتابیس SQLite در حافظه اجرا شدند:

1. **انتقال فازهای چرخه حیات:** تست `test_process_new_closed_candle_lifecycle` اثبات کرد که کندل جدید ابتدا به فاز `PROCESSING` وارد شده و با موفقیت با فاز `COMPLETED` و وضعیت مشخص ذخیره می‌شود.
2. **رد کندل تکراری:** تست `test_reject_duplicate_candle_without_reanalysis` با Mock کردن تابع `FeatureEngineer.transform` اثبات کرد که در چرخه دوم، تابع استخراج ویژگی‌ها دقیقاً صفر بار فراخوانی می‌شود (`call_count == 0`).
3. **رد کندل ناقص موقعیت صفر:** تست `test_reject_forming_candle_via_start_pos_1` تأیید کرد که پارامتر ارسالی به ارائه‌دهنده داده همواره `start_pos=1` است.
4. **رد داده‌های ناقص:** تست `test_invalid_or_incomplete_data` نشان داد با ارسال کمتر از ۵۰ کندل، وضعیت `DATA_INVALID` صادر شده و پایپ‌لاین متوقف می‌شود.
5. **شبیه‌سازی رقابت و تداخل همزمان:** تست `test_concurrent_duplicate_candle_reservation` نشان داد در صورت وجود رکورد رزرو فعال از یک نخ دیگر، Worker بلافاصله متوجه وضعیت `BAR_UNDER_PROCESSING` شده و از اجرای تحلیل صرف‌نظر می‌کند.

---

## ۶. نتایج آزمون اتصال واقعی MetaTrader 5

اتصال به ترمینال متاتریدر ۵ روی ماشین محلی اجرا و تله‌متری زیر استخراج شد:

- مسیر ترمینال: `C:\Program Files\<YourBroker> MT5\terminal64.exe`
- وضعیت اتصال ترمینال به کارگزار: `terminal_info.connected = True`
- شماره حساب معاملاتی: `58033583` (نوع حساب: Demo)
- وضعیت معاملات الگوریتمی: `terminal_info.trade_allowed = False` (غیرفعال بودن در سطح ترمینال تأیید شد)
- وضعیت مظنه نماد XAUUSD:
  - آخرین تیک: `Tick(time=1791590099, bid=4194.34, ask=4194.68, flags=6)`
  - زمان تیک به وقت بین‌المللی: `2026-10-09 23:54:59+00:00`
  - اختلاف زمانی با زمان جاری سیستم: ۳۹.۸ ساعت (جلسه بسته شده در انتهای روز جمعه)
- وضعیت تشخیص بازار:
  - `mt5_status`: `MT5_CONNECTED`
  - `market_status`: `MARKET_CLOSED`
  - علت ثبت‌شده: `Market is closed for the weekend (UTC day=Sunday, tick age=39.8h).`

---

## ۷. تعداد کندل‌های جدید مشاهده‌شده و پردازش‌شده در بازار زنده

در طول بررسی برخط ترمینال متاتریدر ۵:
- تعداد کندل‌های جدید ارائه‌شده توسط کارگزار: **۰ کندل** (بازار طلا تا بازگشایی نشست جهانی در تعطیلی آخر هفته است).
- آخرین کندل بسته ثبت‌شده در متاتریدر: کندل ساعت `23:53:00 UTC` روز ۹ اکتبر ۲۰۲۶ با کلید `XAUUSD_M1_202610092353`.
- کندل باز و ناقص موجود در متاتریدر: کندل ساعت `23:54:00 UTC` که با پایان ساعت معاملاتی ناقص مانده است.
- رفتار سیستم: سیستم با اجرای `start_pos=1`، کندل ناقص ساعت ۲۳:۵۴ را نادیده گرفت و با بررسی دیتابیس متوجه شد کندل ۲۳:۵۳ قبلاً پردازش شده است؛ لذا با وضعیت `NO_NEW_CLOSED_BAR` و تصمیم `HOLD` چرخه را بدون ایجاد تغییر یا ارسال سفارش تکمیل کرد.

---

## ۸. نتایج آزمون Restart و Deduplication

برای بررسی پایداری رفتار سیستم پس از خاموش و روشن شدن (Restart)، دو مرحله تست انجام گرفت:

### آزمون خودکار در حافظه
در تست `test_restart_preserves_deduplication_via_database`:
۱. نمونه اول Worker اجرا شده و یک کندل را تحلیل و ذخیره کرد.
۲. نمونه اول متوقف و نمونه دوم کاملاً مستقل ایجاد گردید.
۳. همان کندل مجدداً در اختیار نمونه دوم قرار گرفت.
۴. نمونه دوم با خواندن دیتابیس وضعیت `NO_NEW_CLOSED_BAR` را بازگرداند و ثابت شد هیچ رکوردی مجدداً درج نشده است.

### آزمون روی فرآیند واقعی سیستم
۱. در فاز Q1 رکورد چرخه `cyc_91f897651c` با کلید `XAUUSD_M1_202610092353` در پایگاه داده اصلی `adaptive_trader.db` ثبت شده بود.
۲. در فاز Q2 سرور و اسکریپت‌ها از نو اجرا شدند.
۳. در اتصال زنده به متاتریدر ۵، آخرین کندل واکشی‌شده همان کندل بود.
۴. سیستم جدید با پیام زیر مانع از تحلیل تکراری شد:
`Latest closed bar XAUUSD_M1_202610092353 has already been analyzed in cycle cyc_91f897651c.`
این نتیجه پایداری کامل Deduplication پس از بازنشانی فرآیند را اثبات کرد.

---

## ۹. وضعیت اتمیک بودن ثبت چرخه و محدودیت‌های باقیمانده

- **تراکنش پایگاه داده:** کلیه مراحل تغییر فاز از `PROCESSING` تا `COMPLETED` در تراکنش‌های مجزا با مدیریت صریح Rollback محافظت می‌شوند.
- **تلاش مجدد پس از خطا:** در صورت بروز خطای پایگاه داده یا وقفه غیرمنتظره، رکورد به فاز `FAILED` تغییر وضعیت می‌دهد و خطای پایگاه داده ضبط می‌شود.
- **محدودیت باقیمانده:** در صورتی که دیسک سیستم به صورت سخت‌افزاری قفل شود یا دسترسی به پایگاه داده SQLite مسدود گردد، Worker وضعیت `DATABASE_WRITE_FAILED` را ثبت کرده و متوقف نمی‌شود تا داده‌های تله‌متری در حافظه زنده باقی بمانند؛ اما سابقه دائمی روی دیسک ثبت نخواهد شد.

---

## ۱۰. نتایج آزمون قطع اتصال و بازیابی (Disconnection & Recovery)

در آزمون `test_mt5_temporary_error_and_reconnection`:
- در چرخه اول، اتصال متاتریدر ۵ قطع شد و تلاش برای اتصال شکست خورد؛ Worker با حفظ حیات وضعیت `MT5_DISCONNECTED` را ثبت کرد و تصمیم به `NONE` تغییر یافت.
- در چرخه دوم، با بازگشت اتصال، متد `connect()` موفق شد و Worker به صورت خودکار به چرخه عادی بازگشت و کندل‌ها را ارزیابی نمود.

در آزمون `test_unexpected_worker_error_and_backoff`:
- بروز خطای غیرمنتظره سیستمی (`Kernel driver error`) بررسی شد.
- Worker متوقف نشد، تعداد خطاهای متوالی (`consecutive_errors`) را ثبت کرد و با اعمال ضریب تاخیر تصاعدی (Exponential Backoff) از اجرای پی‌درپی و مصرف غیرضروری پردازنده جلوگیری به عمل آورد.

---

## ۱۱. نتایج بررسی Event Loop، چندنخی (Threading) و پایگاه داده

- **قفل سراسری MT5:** تمام فراخوانی‌های کتابخانه `MetaTrader5` اکنون تحت `_mt5_lock = threading.RLock()` اجرا می‌شوند. بنابراین تداخل تصادفی بین درخواست‌های ورودی وب‌سرور (نظیر اندپوینت `/status`) و پایش پس‌زمینه ناممکن شده است.
- **آزمون عدم مسدودسازی Event Loop:** در آزمون `test_non_blocking_event_loop_during_worker_cycle`، یک عملیات مسدودکننده ۲۰۰ میلی‌ثانیه‌ای در تابع دریافت داده شبیه‌سازی شد و همزمان ضربان ناهمگام حلقه رویداد (`asyncio.sleep(0.04)`) سنجیده شد. حلقه رویداد در حین اجرای ترد موفق شد ۵ تیک کامل را ثبت کند؛ این نتیجه اثبات کرد که استفاده از `asyncio.to_thread` پاسخگویی رابط کاربری و سرور HTTP را کاملاً حفظ می‌کند.

---

## ۱۲. خطاهای مشاهده‌شده و نحوه رفع آنها

1. **خطای عدم شناسایی نشانگر `@pytest.mark.asyncio`:**
   - *علت:* در محیط تست سیستم پلاگین `pytest-asyncio` نصب نبود و تعریف دستی مارکر باعث توقف جمع‌آوری تست‌ها می‌شد.
   - *راه‌حل:* تست‌های ناهمگام به توابع استاندارد `def test_...` تبدیل شده و در داخل آنها از `asyncio.run()` استفاده شد.
2. **خطای `OperationalError: no such table: emergency_stop_states` در تست نخی:**
   - *علت:* استفاده از دیتابیس `:memory:` ساده بدون استخر اشتراکی در محیط‌های چندنخی باعث می‌شد نخ‌های فرعی پایگاه داده مجزای خالی ببینند.
   - *راه‌حل:* ارتقای فیکسچر تست به `StaticPool` با آرگومان `connect_args={"check_same_thread": False}`.
3. **هم‌پوشانی وضعیت پس از توقف در تست Backoff:**
   - *علت:* استعلام وضعیت `WORKER_EXCEPTION` پس از فراخوانی `stop()` که وضعیت را به `STOPPED` تغییر می‌داد انجام می‌شد.
   - *راه‌حل:* استعلام تله‌متری خطا دقیقاً در حین اجرای تسک و قبل از صدور دستور خاموشی بررسی گردید.

---

## ۱۳. فرامین آزمون و نتایج واقعی

### ۱. اجرای مجموعه آزمون‌های اختصاصی فاز Q2
**فرمان:**
```powershell
python -m pytest backend/tests/unit/test_phase_q2_worker_validation.py -v
```
**نتیجه واقعی:**
```text
backend/tests/unit/test_phase_q2_worker_validation.py::test_process_new_closed_candle_lifecycle PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_reject_duplicate_candle_without_reanalysis PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_reject_forming_candle_via_start_pos_1 PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_invalid_data_triggers_data_invalid_status PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_restart_preserves_deduplication_via_database PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_database_failure_resilience_and_rollback PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_mt5_temporary_error_and_reconnection PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_unexpected_worker_error_and_backoff PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_concurrent_duplicate_candle_reservation PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_graceful_shutdown_releases_resources PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_mt5_connected_vs_data_freshness_separation PASSED
backend/tests/unit/test_phase_q2_worker_validation.py::test_non_blocking_event_loop_during_worker_cycle PASSED
======================= 12 passed in 18.70s =======================
```

### ۲. اجرای کل آزمون‌های پروژه
**فرمان:**
```powershell
python -m pytest backend/tests/ -q
```
**نتیجه واقعی:**
```text
........................................................................ [ 74%]
.........................                                                [100%]
======================= 97 passed in 23.45s =======================
```

---

## ۱۴. تعداد سفارش‌های ارسال‌شده

- تعداد سفارش‌های آزمایشی یا دمو در طول فاز Q2: **۰**
- تعداد سفارش‌های واقعی ارسال‌شده به کارگزار: **۰**
- تعداد موقعیت‌های باز در متاتریدر ۵: **۰**
- وضعیت دروازه دمو: `DEMO_TRADING_ENABLED = False`

عدم ارسال هرگونه سفارش بر اساس گزارش بازرسی دیتابیس و ترمینال متاتریدر ۵ به صورت قطعی تأیید شده است.

---

## ۱۵. جدول خلاصه ارزیابی آزمون‌ها

| آزمون (Test) | محیط (Environment) | وضعیت (Result) | مدرک (Evidence) | یادداشت‌ها (Notes) |
| :--- | :--- | :--- | :--- | :--- |
| چرخه حیات دریافت و تکمیل کندل | Mock / SQLite Memory | **PASS** | `test_process_new_closed_candle_lifecycle` | تأیید فازهای DISCOVERED -> PROCESSING -> COMPLETED |
| پیشگیری از تحلیل مجدد کندل تکراری | Mock / Unit | **PASS** | `test_reject_duplicate_candle_without_reanalysis` | تابع FeatureEngineer صفر بار فراخوانی شد |
| رد کندل در حال تشکیل (موقعیت صفر) | Mock / Unit | **PASS** | `test_reject_forming_candle_via_start_pos_1` | پارامتر `start_pos=1` در کلیه فراخوانی‌ها تأیید شد |
| اعتبارسنجی رد داده‌های نامعتبر | Mock / Unit | **PASS** | `test_invalid_data_triggers_data_invalid_status` | وضعیت `DATA_INVALID` با کندل کمتر از ۵۰ صادر شد |
| حفظ Deduplication در راه‌اندازی مجدد | Mock / Unit | **PASS** | `test_restart_preserves_deduplication_via_database` | کلید `bar_key` در نمونه دوم از دیتابیس خوانده شد |
| پایداری در برابر خطای Commit پایگاه داده | Mock / Unit | **PASS** | `test_database_failure_resilience_and_rollback` | رول‌بک ایمن بدون سقوط Worker اجرا شد |
| قطع موقت و اتصال خودکار مجدد MT5 | Mock / Unit | **PASS** | `test_mt5_temporary_error_and_reconnection` | تفکیک `MT5_DISCONNECTED` و بازیابی به `MT5_CONNECTED` |
| خطای پیش‌بینی‌نشده و تأخیر تصاعدی | Mock / Async | **PASS** | `test_unexpected_worker_error_and_backoff` | فعال شدن مکانیزم Backoff با سقف ۶۰ ثانیه |
| حفاظت از رقابت همزمان برای یک کندل | Mock / Unit | **PASS** | `test_concurrent_duplicate_candle_reservation` | تشخیص وضعیت `BAR_UNDER_PROCESSING` |
| خاموشی کنترل‌شده و آزادسازی منابع | Mock / Async | **PASS** | `test_graceful_shutdown_releases_resources` | فراخوانی قطع ارتباط MT5 و لغو تسک در توقف |
| تفکیک اتصال از کهنگی داده‌های بازار | Mock / Unit | **PASS** | `test_mt5_connected_vs_data_freshness_separation` | پوشش حالات FRESH، NO_RECENT_TICK، CLOSED و STALE |
| پاسخگویی Event Loop در عملیات مسدودکننده | Mock / Async | **PASS** | `test_non_blocking_event_loop_during_worker_cycle` | ثبت ۵ تیک ضربان ناهمگام در حین اجرای همزمان ترد |
| اتصال ترمینال واقعی متاتریدر ۵ | MT5 Live / the configured broker | **PASS** | لاگ اتصال ترمینال و استعلام لاگین ۵۸۰۳۳۵۸۳ | اتصال برقرار و پایدار است |
| دریافت کندل بسته جدید در بازار زنده | MT5 Live Market | **BLOCKED** | تیک جمعه با کهنگی ۳۹.۸ ساعت (`MARKET_CLOSED`) | به دلیل تعطیلی آخر هفته بازارهای جهانی مسدود است |
| جلوگیری از ارسال هرگونه سفارش معاملاتی | Live / Mock | **PASS** | دیتابیس با ۰ رکورد و MT5 با ۰ پوزیشن | تعهد امنیتی صد در صد حفظ شد |

---

## ۱۶. محدودیت‌های محیط، ریسک‌های باقیمانده و پیشنهادهای فاز بعد

### محدودیت‌های محیطی فعلی
- **تعطیلی تقویمی بازار:** اجرای ارزیابی در ساعات تعطیلی بازارهای جهانی طلا (آخر هفته) مانع از مشاهده جریان زنده ورود کندل‌های واقعی جدید گردید. با وجود سلامت کامل منطق نرم‌افزاری و اثبات آن در ۱۲ آزمون، تایید دریافت کندل جدید در بازار واقعی نیازمند پایش در ساعات بازگشایی نشست جهانی است.

### ریسک‌های باقیمانده
1. **اسپرد غیرعادی در زمان بازگشایی نشست:** در دقایق آغازین بازگشایی بازار در یکشنبه شب یا دوشنبه صبح، اسپرد طلا معمولاً جهش شدیدی دارد که فیلتر اسپرد (`MAX_ALLOWED_SPREAD_POINTS = 35`) مانع از صدور سیگنال خواهد شد.
2. **انباشت داده‌های گذشته پس از قطعی طولانی:** اگر ترمینال چند روز خاموش باشد، در اولین اتصال ممکن است تاریخچه ناقص دریافت کند؛ سیستم با فیلتر `count=100, start_pos=1` صرفاً آخرین کندل را بررسی می‌کند و کندل‌های قبل از آن وارد حلقه سیگنال‌دهی نخواهند شد که این رفتار از نظر امنیتی مطلوب است.

### پیشنهادهای فاز بعد
- اجرای یک آزمون پایش زنده به محض شروع نشست رسمی بازار در ساعات اولیه کاری هفته جهت ثبت حداقل ۳ کندل بسته واقعی در تله‌متری زنده.
- بررسی نرخ زمان تاخیر پردازش ویژگی‌ها در بازار با بار واقعی تیک‌ها.
