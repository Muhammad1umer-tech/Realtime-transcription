# Conversational LLM Platform

A real-time conversational platform powered by large language models (LLMs). This project leverages WebSockets for real-time communication and integrates with the Whisper model for transcription and conversational capabilities.

## Table of Contents
- [Conversational LLM Platform](#conversational-llm-platform)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Additional Resources](#additional-resources)

## Project Description

This project aims to build an open-source conversational platform powered by large language models (LLMs). It provides real-time transcription and conversational capabilities using WebSockets for communication between the client and server. The server processes audio data using the Whisper model and returns transcribed text and responses.

## Features

- Real-time audio transcription
- WebSocket-based communication
- Integration with Whisper model for transcription and conversation
- Easy-to-use web interface

## Installation

### Prerequisites

- Python 3.x
- Node.js
- Web browser with WebSocket and MediaStream API support

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/conversational-llm-platform.git
   cd conversational-llm-platform
   ```

2. **Set up the Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install torch whisper
   ```

3. **Set up the Node.js environment:**
   ```bash
   npm install
   ```

## Usage

1. **Start the server:**
   ```bash
   python server/main.py
   ```

2. **Open `client/index.html` in your web browser:**
   ```html:index.html
   startLine: 1
   endLine: 15
   ```

3. **Start and stop the conversation using the buttons on the web page:**
   ```javascript:client/client.js
   startLine: 7
   endLine: 12
   ```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes.
4. Commit your changes: `git add . && git commit -m "Add new feature"`.
5. Push to the branch: `git push origin feature-branch`.
6. Open a pull request against the `main` branch.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Whisper Model](https://github.com/openai/whisper)
- [WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## Additional Resources

- [How to Write a Good README File](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
- [The Ultimate Guide to Writing a Great README.md for Your Project](https://medium.com/@kc_clintone/the-ultimate-guide-to-writing-a-great-readme-md-for-your-project-3d49c2023357)
- [Better README](https://github.com/schultyy/better-readme)

