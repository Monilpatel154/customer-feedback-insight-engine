"""
Dataset Generation Script
Creates a comprehensive, high-quality, realistic customer feedback dataset
for sentiment analysis and aspect-based insight extraction.
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta

def create_feedback_dataset(output_path="data/raw/customer_feedback.csv", num_samples=1800, seed=42):
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define realistic review templates with natural language variations
    # Categories: Electronics, Audio, Wearables, Smart Home, PC Accessories
    categories = ["Electronics", "Audio", "Wearables", "Smart Home", "PC Accessories"]
    
    # Specific aspect feedback pools
    aspect_pools = {
        "Product Quality": {
            "Positive": [
                "The build quality is exceptional, feels sturdy and well-engineered.",
                "High quality materials used, exceeded my expectations completely.",
                "Durability is top notch. Dropped it twice accidentally and no damage at all.",
                "Superb hardware performance and the finish looks very premium.",
                "Outstanding craftsmanship and reliable performance under daily heavy usage.",
                "The screen clarity and speaker output quality are phenomenal.",
                "Solid build, great tactile feedback on buttons, feels like an expensive flagship.",
                "Works flawlessly straight out of the box, zero hardware defects found.",
                "Impressive battery endurance and the internal hardware runs surprisingly cool.",
                "Genuinely surprised by how well built this unit is compared to competitors."
            ],
            "Neutral": [
                "Build quality is acceptable for the price, nothing extraordinary.",
                "Decent materials used, but some plastic parts feel slightly cheap.",
                "It works as advertised, neither great nor bad, average hardware.",
                "Product quality is okay for casual use, but don't expect industrial durability.",
                "Performance is mediocre. Does basic tasks fine but stutters during multitasking.",
                "Standard plastic casing, seems fine so far after two weeks of normal use.",
                "Hardware is average, nothing to write home about but does the job.",
                "Battery life is around 5 hours, which is just about manageable for a workday.",
                "Looks fine aesthetically, though structural rigidity could be improved.",
                "Meets the minimum baseline expectation, but lacks premium refinement."
            ],
            "Negative": [
                "Defective unit received, stopped powering on after only three days.",
                "Terrible build quality, cheap creaking plastic that feels very flimsy.",
                "Overheats dangerously within twenty minutes of moderate usage.",
                "Screen has dead pixels right in the center, completely unacceptable QA.",
                "Severe hardware lag and constant crashes, utterly unusable for work.",
                "Battery drains completely in less than an hour, horrible battery life.",
                "The hinge broke on normal opening, fragile construction and shoddy materials.",
                "Inferior product quality, nowhere close to the advertised technical specs.",
                "Microphone sound is muffled and distorted, completely ruined phone calls.",
                "Buttons became unresponsive after one week, waste of my hard-earned money."
            ]
        },
        "Customer Support": {
            "Positive": [
                "Customer service was incredibly helpful, polite, and resolved my issue within minutes.",
                "Support team went above and beyond to arrange an immediate replacement.",
                "Agent was very knowledgeable, friendly, and guided me step-by-step through setup.",
                "Quick response time via live chat, query was sorted out professionally without delays.",
                "Outstanding warranty assistance, hassle-free exchange processed in 24 hours.",
                "The representative was courteous, listened patiently, and solved the configuration error.",
                "Prompt email support with clear screenshots and instructions, highly impressed.",
                "Technical support team was empathetic and provided a swift warranty claim approval.",
                "Exceptional customer care experience, truly values their consumers.",
                "Support executive followed up the next day to confirm everything was functioning smoothly."
            ],
            "Neutral": [
                "Customer support answered my call, but had to wait on hold for 15 minutes.",
                "The agent answered my question eventually, though they seemed quite rushed.",
                "Support was standard. Took two rounds of emails to get my invoice updated.",
                "Representative gave scripted answers initially, but later clarified my doubt.",
                "Average support interaction, problem was solved but took longer than anticipated.",
                "Help desk reached out after 36 hours, could be faster but resolved the ticket.",
                "The chatbot wasn't helpful, but the human agent who took over was decent.",
                "Got my query resolved, although the transfer between departments was cumbersome.",
                "Support staff was polite, but lacked deep technical knowledge about the device firmware.",
                "Fair experience with help desk, nothing particularly impressive or terrible."
            ],
            "Negative": [
                "Worst customer support imaginable, agent was rude and abruptly disconnected the call.",
                "Ignored my warranty claim emails for two weeks, absolute lack of accountability.",
                "Help desk refused to honor the return policy despite clear photographic evidence.",
                "Kept on hold for over 45 minutes and then call got dropped, hopeless service.",
                "Representative gave completely incorrect troubleshooting steps and refused a refund.",
                "Zero customer support response, automated bot keeps looping without human takeover.",
                "Terrible customer care, treated me like a nuisance when asking for replacement.",
                "Support staff was clueless, kept transferring me between four different departments.",
                "Misleading promises by support executives regarding repair turnaround time.",
                "Awful post-purchase service, once they have your money they stop caring completely."
            ]
        },
        "Delivery & Packaging": {
            "Positive": [
                "Lightning fast delivery, arrived a full day before the estimated date.",
                "Immaculate packaging with heavy protective bubble wrap and tamper-proof seal.",
                "Package arrived in pristine condition, delivery courier was very professional.",
                "Super quick dispatch and real-time live GPS tracking was very accurate.",
                "Eco-friendly yet sturdy packaging, product was fully secured from shocks.",
                "Arrived ahead of scheduled delivery window, safe doorstep drop-off.",
                "Excellent logistics, delivery executive called before arrival to confirm availability.",
                "Flawless unboxing experience, clean premium box design and intact seal.",
                "Prompt shipping, tracking notifications kept me updated at every checkpoint.",
                "Very impressed with how quickly and securely the package was delivered."
            ],
            "Neutral": [
                "Delivery arrived on the promised date, standard courier service.",
                "Packaging was somewhat plain cardboard, but the inner device was safe.",
                "Took five days to arrive, which is standard transit time for my location.",
                "Courier left package with neighbor without informing beforehand, but received safely.",
                "Outer box had slight wrinkles, but internal cushion protected the merchandise.",
                "Delivery was delayed by half a day due to weather, but tracking was updated.",
                "Average shipping speed, nothing exceptional to complain or praise about.",
                "Tracking status didn't update for two days, though parcel arrived on schedule.",
                "Packaging was adequate, minimal cushioning but fortunately no damage.",
                "Delivery person asked for OTP promptly, delivery took the usual standard timeframe."
            ],
            "Negative": [
                "Package arrived completely crushed and soaked in moisture, awful courier handling.",
                "The delivery was very late and box arrived broken with damaged items inside.",
                "Extremely delayed delivery, took more than two weeks past the scheduled arrival date.",
                "The outer box was opened and the product inside was broken and defective.",
                "Courier marked package as delivered when no one actually came, had to chase them for days.",
                "Flimsy packaging with zero protective padding, product arrived with visible dents and broken parts.",
                "Horrible shipping experience, package was sent to the wrong city transit hub twice.",
                "Delivery driver was extremely unprofessional and threw the parcel over the gate.",
                "Package was lost in transit for a week and arrived late with torn packaging.",
                "Disastrous delivery, product casing was cracked and broken due to rough transit impact.",
                "Late delivery ruined a birthday gift surprise, completely unreliable logistics service."
            ]
        },
        "Pricing & Value": {
            "Positive": [
                "Incredible value for money, outperforms products that cost twice as much.",
                "Got this at a great discount, absolutely worth every single penny spent.",
                "Unbeatable price-to-performance ratio in this budget category.",
                "High-end premium features offered at an accessible and reasonable price point.",
                "Best investment I made this month, total bang for your buck.",
                "Very economical pricing without cutting corners on essential functionality.",
                "Competitive price tag, easily the best option available under this budget range.",
                "Fair pricing for such rich features and solid build quality.",
                "Exceptional deal during the sale, thoroughly satisfied with this purchase value.",
                "Offers tremendous cost efficiency, definitely superior to pricier brand alternatives."
            ],
            "Neutral": [
                "Price is reasonable for what you get, neither a steal nor overpriced.",
                "Fairly priced compared to existing market alternatives with similar specs.",
                "Value for money is average, you get what you pay for.",
                "A bit on the higher side, but acceptable given the current market inflation.",
                "Priced appropriately, though competitor brands offer similar features slightly cheaper.",
                "It is decently priced, wait for a sale discount if you want better value.",
                "Cost matches the output quality, straightforward transaction.",
                "Affordable enough for students, though budget compromises are visible.",
                "Just about justifies its price tag, don't expect premium flagship luxury.",
                "Pricing is acceptable, though replacement accessories are somewhat expensive."
            ],
            "Negative": [
                "Absurdly overpriced for such poor functionality and cheap construction.",
                "Complete rip-off, you can get far better alternatives at half the cost.",
                "Grossly overpriced, feels like a cheap ten dollar gadget sold at premium rates.",
                "Regret buying this, zero value for money and totally unjustified pricing.",
                "Hidden subscription fees not mentioned on product page, feels predatory and deceptive.",
                "Extremely expensive maintenance and replacement parts cost more than the device itself.",
                "Overhyped and overpriced, thoroughly disappointed with the value proposition.",
                "Total waste of budget, inferior hardware marked up by aggressive marketing.",
                "Way too expensive for something that lacks standard modern basic features.",
                "Price gouging at its finest, nowhere near worth the steep asking price."
            ]
        },
        "Usability & Interface": {
            "Positive": [
                "The companion app is intuitive, beautifully designed, and very easy to navigate.",
                "Seamless pairing process, took literally 30 seconds to connect and start using.",
                "User interface is clean, responsive, and packed with customizable shortcut options.",
                "Extremely user-friendly interface, even my elderly parents found it easy to operate.",
                "Effortless ergonomics and controls are laid out very sensibly for one-handed operation.",
                "Clean software with zero bloatware and smooth navigation animations.",
                "Setup guide was crystal clear with diagrams, painless initial configuration.",
                "Voice commands and gesture shortcuts work reliably every single time.",
                "Software update was seamless and added really neat personalization widgets.",
                "Pleasant ergonomic design, comfortable to hold and interact with for prolonged hours."
            ],
            "Neutral": [
                "Interface is simple enough, though some menu settings are buried deep.",
                "Companion mobile app works fine, occasionally takes a few seconds to sync data.",
                "Setup is straightforward if you follow the manual, slightly confusing otherwise.",
                "Ergonomics are fine for average hand sizes, might feel bulky for smaller grips.",
                "Software is functional but aesthetics feel somewhat dated and utilitarian.",
                "Initial Bluetooth pairing required a retry, but worked stably afterwards.",
                "App layout is adequate, would appreciate more customization options in future updates.",
                "Controls are responsive, although button labeling could be clearer.",
                "Usability is acceptable, takes a day or two of learning curve to get accustomed.",
                "Basic interface that gets the job done without bells and whistles."
            ],
            "Negative": [
                "Companion app is full of bugs, crashes continuously on opening.",
                "Complicated and frustrating setup process, confusing error codes with no explanation.",
                "Awful user interface design, cluttered screens and counter-intuitive navigation.",
                "Bluetooth connection drops randomly every 5 minutes, infuriating to use.",
                "Touch controls are hyper-sensitive and register phantom touches constantly.",
                "Companion app demands unnecessary location and contact permissions just to operate.",
                "Firmware update bricked the settings menu, requiring complete factory reset.",
                "Horrible ergonomics, uncomfortable to wear or hold after just 10 minutes.",
                "Font size on display is tiny and unreadable under outdoor sunlight.",
                "Clunky UI lag makes standard navigation feel painfully slow and unresponsive."
            ]
        }
    }
    
    # Sentence connectors and modifiers for synthesizing rich multi-sentence feedbacks
    connectors_positive = [
        " Furthermore, overall performance has been rock solid.",
        " Highly recommended to anyone looking for a dependable option.",
        " Truly pleased with this purchase!",
        " Will definitely buy from this brand again.",
        " Five stars without any hesitation."
    ]
    
    connectors_negative = [
        " I strongly urge buyers to stay away from this product.",
        " Will be returning this immediately for a full refund.",
        " Very frustrating experience overall.",
        " Deeply disappointed with this purchase.",
        " Do not waste your time and money on this."
    ]
    
    connectors_mixed = [
        " However, there is definitely room for minor improvements.",
        " On the whole, it is an okay balance between pros and cons.",
        " Still deciding whether to keep it or exchange it.",
        " It works for basic needs but don't expect miracles."
    ]

    records = []
    base_date = datetime(2025, 1, 1)
    
    # Generate balanced & diverse dataset
    sentiments = ["Positive", "Neutral", "Negative"]
    sentiment_weights = [0.45, 0.20, 0.35] # Realistic e-commerce distribution
    
    rating_map = {
        "Positive": [4, 5],
        "Neutral": [3],
        "Negative": [1, 2]
    }
    
    aspect_list = list(aspect_pools.keys())
    
    for i in range(num_samples):
        feedback_id = f"FB-{10001 + i}"
        sentiment = random.choices(sentiments, weights=sentiment_weights)[0]
        rating = random.choice(rating_map[sentiment])
        aspect = random.choice(aspect_list)
        category = random.choice(categories)
        
        # Pick core review sentence
        base_review = random.choice(aspect_pools[aspect][sentiment])
        
        # 40% chance of adding a second sentence (from same or secondary aspect)
        add_extra = random.random()
        if add_extra < 0.40:
            if sentiment == "Positive":
                tail = random.choice(connectors_positive)
            elif sentiment == "Negative":
                tail = random.choice(connectors_negative)
            else:
                tail = random.choice(connectors_mixed)
            full_text = f"{base_review}{tail}"
        elif add_extra < 0.70:
            # Multi-aspect review (e.g. good product but slow shipping)
            secondary_aspect = random.choice([a for a in aspect_list if a != aspect])
            sec_sentiment = random.choice(sentiments)
            sec_sentence = random.choice(aspect_pools[secondary_aspect][sec_sentiment])
            full_text = f"{base_review} Also, regarding {secondary_aspect.lower()}, {sec_sentence.lower()}"
        else:
            full_text = base_review
            
        random_days = random.randint(0, 365)
        random_hours = random.randint(0, 23)
        feedback_date = base_date + timedelta(days=random_days, hours=random_hours)
        
        records.append({
            "feedback_id": feedback_id,
            "category": category,
            "aspect": aspect,
            "rating": rating,
            "sentiment": sentiment,
            "review_text": full_text,
            "timestamp": feedback_date.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Dataset successfully created at '{output_path}' with {len(df)} samples.")
    print("\nClass distribution:")
    print(df["sentiment"].value_counts())
    print("\nAspect distribution:")
    print(df["aspect"].value_counts())
    return df

if __name__ == "__main__":
    create_feedback_dataset()
