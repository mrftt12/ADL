import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { 
  Upload, 
  Brain, 
  Settings, 
  Play, 
  BarChart3,
  CheckCircle,
  ArrowRight,
  Database,
  Layers,
  TrendingUp,
  Download,
  Eye
} from "lucide-react";

export const AutoMLWorkflow = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleConfigureSearchSpace = () => {
    toast({
      title: "Configure Search Space",
      description: "Architecture search configuration dialog would open here",
    });
  };

  const handleDownloadModel = () => {
    toast({
      title: "Download Model",
      description: "Model download would start here",
    });
  };

  const handleViewReport = () => {
    toast({
      title: "View Full Report",
      description: "Detailed analysis report would open here",
    });
  };

  const handleManageDatasets = () => {
    navigate("/datasets");
  };
  const workflowSteps = [
    {
      id: 1,
      title: "Data Upload",
      description: "Upload your dataset and configure preprocessing",
      icon: Upload,
      status: "completed"
    },
    {
      id: 2,
      title: "Architecture Search",
      description: "Neural Architecture Search finds optimal model structure",
      icon: Brain,
      status: "active"
    },
    {
      id: 3,
      title: "Hyperparameter Optimization",
      description: "Automated tuning of learning parameters",
      icon: Settings,
      status: "pending"
    },
    {
      id: 4,
      title: "Model Training",
      description: "Train selected models with optimal configurations",
      icon: Play,
      status: "pending"
    },
    {
      id: 5,
      title: "Evaluation & Results",
      description: "Performance analysis and model comparison",
      icon: BarChart3,
      status: "pending"
    }
  ];

  const getStepColor = (status: string) => {
    switch (status) {
      case "completed": return "text-green-600";
      case "active": return "text-primary";
      case "pending": return "text-muted-foreground";
      default: return "text-muted-foreground";
    }
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case "completed": return CheckCircle;
      case "active": return Brain;
      default: return null;
    }
  };

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h2 className="text-3xl font-bold mb-2">AutoML Workflow</h2>
        <p className="text-muted-foreground">
          Complete automated machine learning pipeline from data to deployment
        </p>
      </div>

      <Tabs defaultValue="workflow" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="workflow">Workflow</TabsTrigger>
          <TabsTrigger value="architecture">Architecture Search</TabsTrigger>
          <TabsTrigger value="hyperparameters">Hyperparameters</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        <TabsContent value="workflow" className="space-y-6">
          <div className="grid gap-6">
            {workflowSteps.map((step, index) => {
              const IconComponent = step.icon;
              const StatusIcon = getStepIcon(step.status);
              
              return (
                <Card key={step.id} className={`transition-all duration-300 ${
                  step.status === "active" ? "shadow-glow border-primary/50" : "shadow-elegant"
                }`}>
                  <CardContent className="flex items-center p-6">
                    <div className="flex items-center gap-4 flex-1">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        step.status === "completed" ? "bg-green-100 text-green-600" :
                        step.status === "active" ? "bg-primary/10 text-primary" :
                        "bg-muted text-muted-foreground"
                      }`}>
                        {StatusIcon ? <StatusIcon className="w-6 h-6" /> : <IconComponent className="w-6 h-6" />}
                      </div>
                      
                      <div className="space-y-1 flex-1">
                        <h3 className={`font-semibold ${getStepColor(step.status)}`}>
                          {step.title}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {step.description}
                        </p>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <Badge variant={
                          step.status === "completed" ? "secondary" :
                          step.status === "active" ? "default" : "outline"
                        }>
                          {step.status}
                        </Badge>
                        
                        {index < workflowSteps.length - 1 && (
                          <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="architecture" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-primary" />
                  Search Space Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Architecture Types</span>
                    <Badge variant="outline">CNN, ResNet, EfficientNet</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Depth Range</span>
                    <Badge variant="outline">3-50 layers</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Width Multiplier</span>
                    <Badge variant="outline">0.5x - 2.0x</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Search Strategy</span>
                    <Badge variant="outline">Progressive ENAS</Badge>
                  </div>
                </div>
                <Button 
                  variant="outline" 
                  className="w-full"
                  onClick={handleConfigureSearchSpace}
                >
                  <Settings className="w-4 h-4" />
                  Configure Search Space
                </Button>
              </CardContent>
            </Card>

            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  Current Best Architecture
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Architecture ID</span>
                    <Badge variant="secondary">#A-47291</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Validation Accuracy</span>
                    <span className="font-semibold text-green-600">89.2%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Parameters</span>
                    <span className="font-semibold">2.1M</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">FLOPs</span>
                    <span className="font-semibold">45.2M</span>
                  </div>
                </div>
                <Button variant="outline" className="w-full">
                  <Eye className="w-4 h-4" />
                  View Architecture Details
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="hyperparameters" className="space-y-6">
          <div className="grid md:grid-cols-3 gap-6">
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle>Learning Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Current Best</span>
                    <span className="font-semibold">0.001</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Search Range</span>
                    <span className="font-semibold">1e-5 to 1e-1</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Strategy</span>
                    <span className="font-semibold">Log-uniform</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle>Batch Size</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Current Best</span>
                    <span className="font-semibold">64</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Search Range</span>
                    <span className="font-semibold">16 to 512</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Strategy</span>
                    <span className="font-semibold">Powers of 2</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle>Optimizer</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Current Best</span>
                    <span className="font-semibold">AdamW</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Search Space</span>
                    <span className="font-semibold">Adam, AdamW, SGD</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Beta1</span>
                    <span className="font-semibold">0.9</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle>Model Performance</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Test Accuracy</span>
                    <span className="font-bold text-lg text-green-600">94.7%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">F1 Score</span>
                    <span className="font-semibold">0.946</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Inference Time</span>
                    <span className="font-semibold">12.3ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Model Size</span>
                    <span className="font-semibold">8.4MB</span>
                  </div>
                </div>
                <Button 
                  variant="gradient" 
                  className="w-full"
                  onClick={handleDownloadModel}
                >
                  <Download className="w-4 h-4" />
                  Download Model
                </Button>
              </CardContent>
            </Card>

            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle>Experiment Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Total Runtime</span>
                    <span className="font-semibold">3h 45m</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Architectures Tested</span>
                    <span className="font-semibold">247</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Hyperparameter Trials</span>
                    <span className="font-semibold">89</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Best Configuration</span>
                    <Badge variant="secondary">Trial #73</Badge>
                  </div>
                </div>
                <Button 
                  variant="outline" 
                  className="w-full"
                  onClick={handleViewReport}
                >
                  <BarChart3 className="w-4 h-4" />
                  View Full Report
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};