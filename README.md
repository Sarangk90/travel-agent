# The Travel Agent

A multi-agent **Travel Assistant** built with [Langgraph](https://github.com/langgraph/langgraph) and [LangChain](https://github.com/langchain-ai/langchain).  
It can search flights and hotels via SerpAPI, orchestrated by a Supervisor agent that hands off to specialized sub-agents.

---

## 🚀 Features

- **Supervisor Agent**  
  Greets the user, understands intent, and routes to the appropriate sub-agent (flights or hotels).

- **Flights Advisor Agent**  
  - Search round-trip or one-way flights by IATA codes  
  - Enforces date validation (no past dates, correct format)  
  - Fetches detailed itineraries with pricing, airlines, schedule, layovers  

- **Hotel Advisor Agent**  
  - Search hotels by location, check-in/check-out dates  
  - Sort by rating, price, popularity, etc.  
  - Validate date ranges and room/guest counts  

- **Handoff Tool**  
  Seamlessly transfer conversation context between agents.

---

## 📂 Project Structure

```

sarangk90-travel-agent/
├── langgraph.json            # Langgraph entrypoint & config
├── main.py                   # CLI launcher
├── requirements.txt          # Python dependencies
└── app/
├── graph.py              # StateGraph definition & MemorySaver
├── agents/
│   ├── flights\_advisor\_agent.py
│   ├── hotel\_advisor\_agent.py
│   └── supervisor\_agent.py
└── tools/
└── handoff\_tool.py   # make\_handoff\_tool factory

````

---

## ⚙️ Prerequisites

- **Python** 3.11  
- A SerpAPI key: [https://serpapi.com/](https://serpapi.com/)  
- An OpenAI API key: [https://platform.openai.com/](https://platform.openai.com/)

---

## 🛠 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/sarangk90-travel-agent.git
   cd sarangk90-travel-agent
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   Copy `.env.example` to `.env` (or create a new `.env`) and set:

   ```dotenv
   SERPAPI_API_KEY=your_serpapi_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

---

## ▶️ Usage

Run the CLI launcher:

```bash
python main.py
```

You’ll be prompted:

```
Enter your message:
```

Example interaction:

```
Enter your message: Hi there!
supervisor: Hello! I’m your travel assistant. I can help with:
• Flight searches  
• Hotel recommendations  

Which would you like to do today?
```

From there, follow the prompts to search flights or hotels.

---

## 🧱 Architecture Overview

1. **main.py**

   * Loads `.env`, initializes a UUID thread
   * Listens for user input and streams responses from the Langgraph `graph`

2. **app/graph.py**

   * Builds a `StateGraph` with nodes: `supervisor`, `flights_advisor`, `hotel_advisor`, and `human`
   * Uses `MemorySaver` to persist conversation state

3. **Agents**

   * **Supervisor Agent**
     *Offers options and routes to sub-agents*
   * **Flights Advisor**
     *Validates inputs & calls `find_flights` tool (SerpAPI + React agent)*
   * **Hotel Advisor**
     *Validates inputs & calls `get_hotel_recommendations` tool*

4. **Handoff Tool**

   * Custom tool that returns a `Command` to switch active agent

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -m "Add XYZ"`)
4. Push to your branch (`git push origin feature/XYZ`)
5. Open a Pull Request!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
