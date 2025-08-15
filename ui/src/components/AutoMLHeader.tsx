import { Button } from "@/components/ui/button";
import { Brain, Target, Cog, Zap } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { CreateProjectDialog } from "@/components/CreateProjectDialog";

export const AutoMLHeader = () => {
  const navigate = useNavigate();
  return (
    <div className="relative overflow-hidden bg-gradient-background border-b">
      <div className="absolute inset-0 bg-gradient-secondary opacity-10 animate-gradient-shift bg-[length:200%_200%]"></div>
      
      <div className="relative container mx-auto px-6 py-16">
        <div className="text-center space-y-6">
          <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium">
            <Brain className="w-4 h-4" />
            AutoML Framework
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-primary bg-clip-text text-transparent animate-float">
            Automated Deep Learning
            <span className="block text-foreground mt-2">Made Simple</span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Harness the power of automated machine learning with Neural Architecture Search, 
            hyperparameter optimization, and intelligent model selection—all without the complexity.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-6">
            <CreateProjectDialog />
            <Button 
              variant="outline" 
              size="xl" 
              className="min-w-48"
              onClick={() => navigate("/examples")}
            >
              <Target className="w-5 h-5" />
              View Examples
            </Button>
          </div>
          
          <div className="grid md:grid-cols-4 gap-6 mt-16 text-center">
            <div className="space-y-3">
              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mx-auto">
                <Brain className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold">Neural Architecture Search</h3>
              <p className="text-sm text-muted-foreground">
                Automatically discover optimal network architectures
              </p>
            </div>
            
            <div className="space-y-3">
              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mx-auto">
                <Cog className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold">Hyperparameter Optimization</h3>
              <p className="text-sm text-muted-foreground">
                Intelligent search for best model parameters
              </p>
            </div>
            
            <div className="space-y-3">
              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mx-auto">
                <Zap className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold">Automated Training</h3>
              <p className="text-sm text-muted-foreground">
                End-to-end model training with minimal intervention
              </p>
            </div>
            
            <div className="space-y-3">
              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mx-auto">
                <Target className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold">Model Evaluation</h3>
              <p className="text-sm text-muted-foreground">
                Comprehensive performance analysis and comparison
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};