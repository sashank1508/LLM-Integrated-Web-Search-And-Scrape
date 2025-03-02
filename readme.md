# LLM-Integrated Web Search & Scrape

## 🚀 Overview

This project is an **AI-driven web scraper and search query answering system** that automates information retrieval from the web. It performs the following:

✔ **Extracts URLs & Queries** from user input  
✔ **Finds relevant webpages** via Google/DuckDuckGo search  
✔ **Scrapes content** from the top-ranked websites  
✔ **Uses OpenAI's GPT to analyze and extract answers** from the scraped content  
✔ **Handles multiple iterations** to refine results and fetch more data when necessary
✔ **Goes multiple levels deep** into a website if needed to find answers
✔ **Delivers a structured final answer** based on all collected information  

This system is useful for **news analysis, research automation, fact-checking, and competitive intelligence.**

---

## 📜 Features

✔ **Automated URL Extraction** – Detects and processes URLs from input text  
✔ **Google/DuckDuckGo Search Integration** – Finds top search results for queries  
✔ **Smart Web Scraping** – Extracts data using **aiohttp** and **BeautifulSoup**  
✔ **AI-Driven Query Resolution** – Uses **GPT-4o-mini** to analyze and extract meaningful responses  
✔ **Iterative Query Refinement** – Moves to additional pages if the answer is not found  
✔ **Recursive Website Navigation** – Crawls deeper into a site to locate relevant information
✔ **Customizable Skip Lists** – Prevents specific websites from being scraped  
✔ **Structured Logging & Error Handling** – Keeps track of all operations efficiently  

---

## 🛠️ Tech Stack

| Technology       | Purpose                                                        |
|-----------------|----------------------------------------------------------------|
| **Python**      | Core programming language for automation & AI processing      |
| **aiohttp**     | Asynchronous HTTP client for web scraping                     |
| **BeautifulSoup** | Parses and extracts useful data from scraped HTML pages     |
| **Selenium (Optional)** | Can be used for more dynamic web scraping scenarios    |
| **Google/DuckDuckGo Search** | Fetches relevant search results from the web     |
| **OpenAI GPT API** | AI-powered natural language processing & response generation |
| **AsyncIO**     | Handles asynchronous execution for efficiency                 |
| **dotenv**      | Manages API keys and credentials securely                      |
| **Logging**     | Tracks script execution and error handling                     |

---

## 📦 System Workflow

1️⃣ **Extract URL & Query from Input**  
2️⃣ **If URL is missing, use Google/DuckDuckGo search to find relevant pages**  
3️⃣ **Scrape the top-ranked website's content**  
4️⃣ **Send scraped content + query to GPT for analysis**  
5️⃣ **If answer is found, return it; otherwise, fetch the next relevant page**  
6️⃣ **If required, navigate deeper into the website to find more specific answers**
7️⃣ **Repeat until an answer is found or all sources are exhausted**  
8️⃣ **Deliver a structured final answer**  

---

## 🏗️ Setup & Installation

### **🔹 Prerequisites**

- Python 3.8+
- Google Chrome & ChromeDriver installed (if using Selenium)
- OpenAI API key

### **🔹 Clone the Repository**

```sh
git clone https://github.com/sashank1508/LLM-Integrated-Web-Search-And-Scrape
cd LLM-Integrated-Web-Search-And-Scrape
```

### **🔹 Create a Virtual Environment**

```sh
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### **🔹 Install Dependencies**

```sh
pip install -r requirements.txt
```

### **🔹 Configure Environment Variables**

Create a `.env` file in the project directory and add:

```ini
OPENAI_API_KEY=your-openai-api-key
```

### **🔹 Run the Notebook**

---

## ⚡ How It Works

1. **Extracts Data from User Input**
   - Identifies URLs and questions automatically
   - If no URL is found, uses Google/DuckDuckGo to find relevant pages

2. **Scrapes Content from the Web**
   - Fetches HTML content from target URLs
   - Cleans & processes relevant text

3. **Uses AI for Analysis**
   - Sends content + query to OpenAI GPT for context-aware answers
   - If answer is not found, fetches next best webpage
   - If necessary, navigates deeper into the website

4. **Returns the Most Accurate Answer**
   - Combines multiple sources if needed
   - Filters out irrelevant data

---

## 📊 Example Output 1

```plaintext
User Input: "What are the latest updates on the KIIT case?"

Base URL: KIIT Case
Query: What are the latest updates on the KIIT case?

Attempt 1:
Scraping Content From URL: https://www.firstpost.com/explainers/kiit-suicide-row

Answer Found:
A 20-year-old Nepali student at KIIT, was found dead in her hostel room on February 16, 2025. Her family and friends allege she was harassed by a fellow student, who has been arrested on charges of abetment of suicide. The Odisha government has initiated a probe, and KIIT has apologized for its handling of the situation.

Attempt 2:
Scraping Content From URL: https://en.wikipedia.org/wiki/KIIT_incident_2025

Answer Found:
On 16 February 2025, A Nepali B.Tech student at KIIT, was found dead in her hostel in a suspected suicide. Protests erupted among students alleging harassment and institutional negligence. The university ordered all Nepali students to vacate, leading to further unrest. A male student was arrested in connection with the case, and investigations are ongoing, with the Nepali Embassy intervening to ensure a thorough inquiry. The situation remains tense as authorities address the concerns raised by the student community.

Final Answer: The latest updates on the KIIT case involve the tragic death of a Nepali student, leading to protests, an ongoing investigation, and university reforms to address safety concerns.
```

---

## 📊 Example Output 2

```plaintext
User Input: "Find the email of Head of Department of Computer Science and Engineering at IIT Roorkee."

Attempt 1: Scraping content from URL: https://iitr.ac.in
Answer: Not Found
Next URL: https://iitr.ac.in/Departments/index.html
Navigating to: https://iitr.ac.in/Departments/index.html

Attempt 2: Scraping content from URL: https://iitr.ac.in/Departments/index.html
Answer: Not Found
Next URL: https://iitr.ac.in/Departments/Computer%20Science%20and%20Engineering%20Department/index.html
Navigating to: https://iitr.ac.in/Departments/Computer%20Science%20and%20Engineering%20Department/index.html

Attempt 3: Scraping content from URL: https://iitr.ac.in/Departments/Computer%20Science%20and%20Engineering%20Department/index.html
Answer: csed@iitr.ac.in
Next URL: None

Final Answer: csed@iitr.ac.in
```

---

## 🔧 Future Enhancements

🔹 **Improve Web Scraping** – Handle JavaScript-heavy websites with Selenium  
🔹 **Expand Search Engine Support** – Include Bing & Yandex search integration  
🔹 **Enhance Prompt Engineering** – Optimize AI responses for better accuracy  
🔹 **Store Results in a Database** – Build a history of previous queries  
🔹 **Deploy as a Web App** – Provide a user-friendly UI for easy querying  

---

## 📝 Author

👨‍💻 **Sashank Yendrapati**  
📧 Email: <sashank1.y@gmail.com>  
🔗 LinkedIn: [Sashank's Profile](https://www.linkedin.com/in/sashank-yendrapati-358498a5/)  
📁 GitHub: [sashank1508](https://github.com/sashank1508)

---
