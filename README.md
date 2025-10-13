# CyberSafe - Cyberbullying Text Classifier

A comprehensive web-based cyberbullying detection application built for college project demonstration.

## Project Overview

CyberSafe is an intelligent text classification system that uses advanced pattern recognition algorithms to detect cyberbullying content in text messages. The application provides real-time analysis with detailed feedback about potential harmful content.

## Features

### Core Functionality
- **Real-time Text Analysis**: Instant classification of input text
- **Multi-category Detection**: Identifies threats, harassment, discrimination, exclusion, and offensive language
- **Confidence Scoring**: Provides percentage-based confidence levels
- **Detailed Results**: Shows specific problematic patterns and categorization
- **Analysis History**: Tracks previous analyses with timestamps
- **Statistics Dashboard**: Displays usage statistics and trends

### User Interface
- **Modern Design**: Clean, professional interface with responsive layout
- **Intuitive Navigation**: Easy-to-use forms and controls
- **Visual Feedback**: Color-coded results and smooth animations
- **Accessibility**: Proper labeling and keyboard navigation support

### Technical Implementation
- **Client-side Processing**: All analysis performed locally for privacy
- **Rule-based Algorithm**: Comprehensive pattern matching system
- **Local Storage**: Persistent history and statistics storage
- **Export Functionality**: CSV export of analysis results
- **Keyboard Shortcuts**: Quick access controls (Ctrl+Enter to analyze, Ctrl+L to clear)

## Technologies Used

- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with custom properties and responsive design
- **JavaScript (ES6+)**: Core functionality and DOM manipulation
- **Local Storage API**: Data persistence
- **Responsive Design**: Mobile-first approach

## File Structure

```
cyberbullying-detector/
├── index.html          # Main HTML structure
├── style.css           # Comprehensive styling
├── app.js              # Core JavaScript functionality
└── README.md           # Project documentation
```

## Installation & Usage

1. **Download**: Extract all files to a local directory
2. **Open**: Open `index.html` in any modern web browser
3. **Analyze**: Enter text in the textarea and click "Analyze Text"
4. **Review**: Examine the detailed results and recommendations

## Algorithm Details

### Detection Categories

1. **Threats/Violence**: Direct or implied threats of harm
2. **Harassment**: Repeated hostile or aggressive behavior  
3. **Discrimination**: Content targeting protected characteristics
4. **Exclusion**: Language designed to isolate or ostracize
5. **Offensive Language**: Profanity and derogatory terms
6. **Sexual Harassment**: Inappropriate sexual content

### Scoring System

- **Offensive Words**: 2 points each
- **Threats**: 5 points each (highest priority)
- **Harassment Patterns**: 4 points each
- **Exclusion Language**: 3 points each
- **Discriminatory Content**: 4 points each
- **Context Modifiers**: Additional points for personal targeting, intensifiers, excessive caps, and punctuation

### Classification Threshold

- **Threshold**: 3 points minimum for cyberbullying classification
- **Confidence**: Calculated as percentage of maximum possible score
- **Severity Levels**: None (0-2), Low (3-5), Medium (6-9), High (10+)

## Privacy & Security

- **Local Processing**: All text analysis performed client-side
- **No Data Transmission**: No information sent to external servers
- **Browser Storage**: History stored locally in browser's localStorage
- **Data Control**: Users can clear history at any time

## Educational Value

This project demonstrates:

- **Text Processing**: Pattern matching and natural language analysis
- **Web Development**: Modern HTML, CSS, and JavaScript techniques
- **User Experience**: Intuitive interface design and accessibility
- **Data Visualization**: Statistics and results presentation
- **Software Engineering**: Modular code organization and documentation

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Future Enhancements

- Machine learning model integration
- Multi-language support
- Advanced context analysis
- API integration capabilities
- Bulk text processing
- Custom pattern configuration

## License

This project is created for educational purposes as a college assignment.

## Contact

For questions or suggestions regarding this college project, please contact the development team.

---

**Note**: This application is designed for educational and demonstration purposes. For production use, consider implementing additional security measures and comprehensive testing.
