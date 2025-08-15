import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AppHeader } from "@/components/AppHeader";
import { useNavigate } from "react-router-dom";
import { 
  Image, 
  MessageSquare, 
  TrendingUp, 
  Clock, 
  Database, 
  Brain,
  Target,
  ArrowRight,
  Download,
  Eye
} from "lucide-react";

interface ExampleProject {
  id: string;
  title: string;
  description: string;
  category: string;
  modelType: string;
  dataset: string;
  accuracy: number;
  sampleCount: number;
  complexity: 'Beginner' | 'Intermediate' | 'Advanced';
  estimatedTime: string;
  tags: string[];
  previewImage?: string;
}

const exampleProjects: ExampleProject[] = [
  {
    id: "cifar10-classification",
    title: "CIFAR-10 Image Classification",
    description: "Train a CNN to classify images into 10 categories using automated architecture search. Perfect for learning computer vision fundamentals.",
    category: "Computer Vision",
    modelType: "CNN Classification",
    dataset: "CIFAR-10",
    accuracy: 94.2,
    sampleCount: 60000,
    complexity: "Beginner",
    estimatedTime: "30-45 min",
    tags: ["Computer Vision", "Classification", "CNN", "AutoML"]
  },
  {
    id: "sentiment-analysis",
    title: "Movie Review Sentiment Analysis",
    description: "Analyze movie reviews to determine positive or negative sentiment using transformer-based models with hyperparameter optimization.",
    category: "Natural Language Processing",
    modelType: "Transformer",
    dataset: "IMDB Reviews",
    accuracy: 92.8,
    sampleCount: 50000,
    complexity: "Intermediate",
    estimatedTime: "45-60 min",
    tags: ["NLP", "Sentiment", "Transformers", "Text Classification"]
  },
  {
    id: "stock-prediction",
    title: "Stock Price Forecasting",
    description: "Predict future stock prices using LSTM networks with automated feature engineering and hyperparameter tuning.",
    category: "Time Series",
    modelType: "LSTM",
    dataset: "S&P 500 Data",
    accuracy: 87.5,
    sampleCount: 25000,
    complexity: "Advanced",
    estimatedTime: "60-90 min",
    tags: ["Time Series", "LSTM", "Finance", "Forecasting"]
  },
  {
    id: "object-detection",
    title: "Vehicle Detection",
    description: "Detect and classify vehicles in traffic images using YOLO-based architecture search for real-time applications.",
    category: "Computer Vision",
    modelType: "YOLO",
    dataset: "Traffic Dataset",
    accuracy: 89.1,
    sampleCount: 15000,
    complexity: "Advanced",
    estimatedTime: "90-120 min",
    tags: ["Object Detection", "YOLO", "Computer Vision", "Real-time"]
  },
  {
    id: "customer-churn",
    title: "Customer Churn Prediction",
    description: "Predict customer churn using tabular data with automated feature selection and ensemble methods.",
    category: "Business Analytics",
    modelType: "Ensemble",
    dataset: "Customer Data",
    accuracy: 91.3,
    sampleCount: 10000,
    complexity: "Intermediate",
    estimatedTime: "30-45 min",
    tags: ["Tabular Data", "Classification", "Business", "Ensemble"]
  },
  {
    id: "speech-recognition",
    title: "Speech Command Recognition",
    description: "Recognize spoken commands using CNN-RNN hybrid architectures with automated preprocessing pipelines.",
    category: "Audio Processing",
    modelType: "CNN-RNN",
    dataset: "Speech Commands",
    accuracy: 93.7,
    sampleCount: 105000,
    complexity: "Advanced",
    estimatedTime: "75-90 min",
    tags: ["Audio", "Speech", "CNN-RNN", "Commands"]
  }
];

const Examples = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const navigate = useNavigate();

  const categories = ["All", ...Array.from(new Set(exampleProjects.map(p => p.category)))];
  
  const filteredProjects = selectedCategory === "All" 
    ? exampleProjects 
    : exampleProjects.filter(p => p.category === selectedCategory);

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'Beginner': return 'bg-green-500/10 text-green-600 border-green-500/20';
      case 'Intermediate': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      case 'Advanced': return 'bg-red-500/10 text-red-600 border-red-500/20';
      default: return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Computer Vision': return <Image className="w-5 h-5" />;
      case 'Natural Language Processing': return <MessageSquare className="w-5 h-5" />;
      case 'Time Series': return <TrendingUp className="w-5 h-5" />;
      case 'Business Analytics': return <Target className="w-5 h-5" />;
      case 'Audio Processing': return <Database className="w-5 h-5" />;
      default: return <Brain className="w-5 h-5" />;
    }
  };

  const handleUseTemplate = (example: ExampleProject) => {
    // Navigate to create project with pre-filled data
    navigate('/create-project', { 
      state: { 
        template: {
          name: example.title,
          description: example.description,
          modelType: example.modelType.toLowerCase().replace(/[^a-z]/g, '_'),
          datasetName: example.dataset
        }
      }
    });
  };

  return (
    <div className="min-h-screen bg-gradient-background">
      <AppHeader />
      
      <div className="container mx-auto px-6 py-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Example Projects</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Explore pre-configured AutoML projects to jumpstart your machine learning journey. 
            Each example includes optimized architectures and hyperparameters.
          </p>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {categories.map((category) => (
            <Button
              key={category}
              variant={selectedCategory === category ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedCategory(category)}
              className="min-w-fit"
            >
              {category !== "All" && getCategoryIcon(category)}
              {category}
            </Button>
          ))}
        </div>

        {/* Examples Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {filteredProjects.map((example) => (
            <Card key={example.id} className="hover:shadow-lg transition-all duration-300 group">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getCategoryIcon(example.category)}
                    <Badge variant="outline" className="text-xs">
                      {example.category}
                    </Badge>
                  </div>
                  <Badge className={getComplexityColor(example.complexity)}>
                    {example.complexity}
                  </Badge>
                </div>
                
                <CardTitle className="text-lg line-clamp-2 group-hover:text-primary transition-colors">
                  {example.title}
                </CardTitle>
                <CardDescription className="line-clamp-3">
                  {example.description}
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <TrendingUp className="w-3 h-3" />
                      Accuracy
                    </div>
                    <div className="font-medium text-green-600">{example.accuracy}%</div>
                  </div>
                  
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      Time
                    </div>
                    <div className="font-medium">{example.estimatedTime}</div>
                  </div>
                  
                  <div className="col-span-2">
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <Database className="w-3 h-3" />
                      Dataset
                    </div>
                    <div className="font-medium">{example.dataset} ({example.sampleCount.toLocaleString()} samples)</div>
                  </div>
                </div>
                
                <div className="text-sm">
                  <div className="text-muted-foreground">Model Type</div>
                  <div className="font-medium">{example.modelType}</div>
                </div>
                
                <div className="flex flex-wrap gap-1">
                  {example.tags.slice(0, 3).map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {example.tags.length > 3 && (
                    <Badge variant="secondary" className="text-xs">
                      +{example.tags.length - 3}
                    </Badge>
                  )}
                </div>
                
                <div className="flex gap-2 pt-2">
                  <Button 
                    size="sm" 
                    variant="outline" 
                    className="flex-1"
                    onClick={() => {
                      // TODO: Implement preview functionality
                      navigate(`/example/${example.id}`);
                    }}
                  >
                    <Eye className="w-3 h-3" />
                    Preview
                  </Button>
                  <Button 
                    size="sm" 
                    variant="gradient" 
                    className="flex-1"
                    onClick={() => handleUseTemplate(example)}
                  >
                    <Download className="w-3 h-3" />
                    Use Template
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredProjects.length === 0 && (
          <div className="text-center py-12">
            <Brain className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No examples found</h3>
            <p className="text-muted-foreground">
              Try selecting a different category to see more examples.
            </p>
          </div>
        )}

        <div className="text-center mt-12">
          <Button 
            variant="outline" 
            size="lg" 
            onClick={() => navigate("/")}
          >
            <ArrowRight className="w-4 h-4 rotate-180" />
            Back to Dashboard
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Examples;