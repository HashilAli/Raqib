# LinkedIn post drafts

Replace `[LINK]` with your Streamlit or Hugging Face URL and `[REPO]` with the GitHub link. Attach 2–3 screenshots: the control room with the overfill alert, the what-if table, and the plume map.

## العربية

🛡️ «رقيب»: نموذج أولي لتوأم رقمي يتنبأ بحوادث المصافي ومحطات المنتجات قبل وقوعها

معظم الحوادث الكبرى تسبقها إشارات ضعيفة: مقياس مستوى يتوقف بصمت، اهتزاز يرتفع ببطء، فرق صغير في موازنة خط الأنابيب. المشكلة أن هذه الإشارات موزعة على أنظمة منفصلة ولا يربطها أحد في الوقت المناسب.

بنيت نموذجاً أولياً مفتوح المصدر يعمل على محطة افتراضية:
🔁 توأم رقمي يقارن كل قراءة بما يجب أن تكون عليه حسب الفيزياء، فيكشف مقياس المستوى العالق بعد نحو 30 دقيقة، وقبل الفيضان بنحو 28 ساعة في المحاكاة (نمط حادثة Buncefield)
🧠 نماذج تعلم آلة لستة أنماط فشل: فيضان الخزان، تسرب الخط، فشل المضخة مع الزمن المتبقي، انقطاع الكهرباء، سحابة الأبخرة، والتلاعب السيبراني
🧪 محرك «ماذا لو» يجرب الحلول على التوأم ويرتبها قبل عرضها على المشغل
📋 جدول FMEA بترميز ISO 14224، ومسودات أوامر عمل للصيانة
🔐 تصميم آمن: قراءة فقط عبر بوابة أحادية الاتجاه، ولا يلمس نظام السلامة
🗣️ واجهة عربية وإنجليزية، وكل تنبيه يأتي مع سببه

المبدأ: النظام مستشار لا متحكم، والقرار يبقى للمشغل.

البيانات اصطناعية بالكامل، والهدف عرض الفكرة وفتح النقاش مع مهندسي العمليات والموثوقية والأمن السيبراني.

جربه هنا: [LINK]
الكود: [REPO]

أرحب بملاحظاتكم، خصوصاً من الزملاء في غرف التحكم والصيانة.

#التوأم_الرقمي #الذكاء_الاصطناعي #السلامة_الصناعية #النفط_والغاز #عمان #DigitalTwin #PredictiveMaintenance #ProcessSafety

## English

🛡️ RAQIB: an open-source digital twin that predicts terminal and refinery incidents before they happen

Most major incidents announce themselves with weak signals: a level gauge that quietly stops moving, vibration creeping up, a small imbalance on a pipeline. Those signals sit in separate systems and nobody connects them in time.

I built an open-source prototype on a simulated product terminal:
🔁 A digital twin that checks every reading against the physics. In the simulation it flags a stuck level gauge within about 30 minutes, roughly 28 hours before the tank would overflow (the Buncefield pattern)
🧠 Machine-learning models for six failure modes: tank overfill, pipeline leak, pump failure with remaining useful life, power loss, vapour cloud, and cyber manipulation
🧪 A what-if engine that tests each fix on the twin and ranks it before the operator sees it
📋 An ISO 14224 FMEA and draft maintenance work orders
🔐 Secure by design: read-only through a one-way gateway, never touches the safety system
🗣️ Arabic and English interface, and every alert explains why

The principle: an advisor, not a controller. The operator decides.

All data is synthetic. The goal is to show the idea and start a conversation with process, reliability and OT security engineers.

Try it: [LINK]
Code: [REPO]

Feedback welcome, especially from people who work in control rooms and maintenance.

#DigitalTwin #AI #ProcessSafety #PredictiveMaintenance #OilAndGas #OTSecurity #Oman
