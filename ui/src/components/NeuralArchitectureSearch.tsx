import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { 
  Brain, 
  Layers, 
  Play, 
  Pause, 
  RotateCcw, 
  TrendingUp,
  Zap,
  Clock,
  Activity
} from "lucide-react";

interface Architecture {
  id: string;
  name: string;
  layers: string[];
  parameters: number;
  flops: number;
  accuracy: number;
  searchIteration: number;
}

interface NASConfig {
  searchSpace: string[];
  maxIterations: number;
  objectiveMetric: string;
}

export const NeuralArchitectureSearch = () => {
  const [isSearching, setIsSearching] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [architectures, setArchitectures] = useState<Architecture[]>([]);
  const [bestArchitecture, setBestArchitecture] = useState<Architecture | null>(null);
  const [searchProgress, setSearchProgress] = useState(0);
  const { toast } = useToast();

  const defaultConfig: NASConfig = {
    searchSpace: ["Conv2D", "BatchNorm", "ReLU", "MaxPool", "Dense", "Dropout"],
    maxIterations: 50,
    objectiveMetric: "accuracy"
  };

  // Simulate architecture generation
  const generateRandomArchitecture = (iteration: number): Architecture => {
    const layerTypes = defaultConfig.searchSpace;
    const numLayers = Math.floor(Math.random() * 8) + 3; // 3-10 layers
    const layers: string[] = [];
    
    for (let i = 0; i < numLayers; i++) {
      const layerType = layerTypes[Math.floor(Math.random() * layerTypes.length)];
      layers.push(layerType);
    }

    const parameters = Math.floor(Math.random() * 5000000) + 100000; // 100K-5M params
    const flops = Math.floor(Math.random() * 10000000) + 1000000; // 1M-10M FLOPs
    const baseAccuracy = 0.85 + Math.random() * 0.1; // 85-95% base
    const iterationBonus = Math.min(iteration * 0.001, 0.05); // Improvement over time
    const accuracy = Math.min(baseAccuracy + iterationBonus, 0.98);

    return {
      id: `arch_${iteration}_${Date.now()}`,
      name: `Architecture ${iteration}`,
      layers,
      parameters,
      flops,
      accuracy: parseFloat(accuracy.toFixed(4)),
      searchIteration: iteration
    };
  };

  const startSearch = () => {
    if (isSearching) return;
    
    setIsSearching(true);
    setCurrentIteration(0);
    setArchitectures([]);
    setBestArchitecture(null);
    setSearchProgress(0);

    toast({
      title: "NAS Started",
      description: "Neural Architecture Search is now running",
    });

    // Simulate search iterations
    const searchInterval = setInterval(() => {
      setCurrentIteration(prev => {
        const newIteration = prev + 1;
        const progress = (newIteration / defaultConfig.maxIterations) * 100;
        setSearchProgress(progress);

        // Generate new architecture
        const newArch = generateRandomArchitecture(newIteration);
        
        setArchitectures(prevArchs => {
          const updatedArchs = [...prevArchs, newArch];
          
          // Update best architecture
          setBestArchitecture(currentBest => {
            if (!currentBest || newArch.accuracy > currentBest.accuracy) {
              if (currentBest) {
                toast({
                  title: "New Best Architecture!",
                  description: `Found architecture with ${(newArch.accuracy * 100).toFixed(2)}% accuracy`,
                });
              }
              return newArch;
            }
            return currentBest;
          });

          return updatedArchs.slice(-10); // Keep last 10 architectures
        });

        // Stop search when max iterations reached
        if (newIteration >= defaultConfig.maxIterations) {
          setIsSearching(false);
          clearInterval(searchInterval);
          toast({
            title: "NAS Completed",
            description: `Search completed after ${newIteration} iterations`,
          });
        }

        return newIteration;
      });
    }, 500); // New architecture every 500ms for demo
  };

  const pauseSearch = () => {
    setIsSearching(false);
    toast({
      title: "NAS Paused",
      description: "Architecture search has been paused",
    });
  };

  const resetSearch = () => {
    setIsSearching(false);
    setCurrentIteration(0);
    setArchitectures([]);
    setBestArchitecture(null);
    setSearchProgress(0);
    toast({
      title: "NAS Reset",
      description: "Search state has been reset",
    });
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            Neural Architecture Search
          </CardTitle>
          <CardDescription>
            Automated discovery of optimal neural network architectures
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{currentIteration}</div>
              <div className="text-sm text-muted-foreground">Iterations</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {architectures.length}
              </div>
              <div className="text-sm text-muted-foreground">Architectures Tested</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {bestArchitecture ? `${(bestArchitecture.accuracy * 100).toFixed(2)}%` : 'N/A'}
              </div>
              <div className="text-sm text-muted-foreground">Best Accuracy</div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Search Progress</span>
              <span>{searchProgress.toFixed(1)}%</span>
            </div>
            <Progress value={searchProgress} className="h-2" />
          </div>

          <div className="flex gap-2">
            {!isSearching ? (
              <Button onClick={startSearch} variant="gradient">
                <Play className="w-4 h-4" />
                Start Search
              </Button>
            ) : (
              <Button onClick={pauseSearch} variant="outline">
                <Pause className="w-4 h-4" />
                Pause Search
              </Button>
            )}
            <Button onClick={resetSearch} variant="outline">
              <RotateCcw className="w-4 h-4" />
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Best Architecture */}
      {bestArchitecture && (
        <Card className="border-green-500/20 bg-green-500/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-green-600">
              <TrendingUp className="w-5 h-5" />
              Best Architecture Found
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-muted-foreground">Accuracy</div>
                <div className="text-lg font-bold text-green-600">
                  {(bestArchitecture.accuracy * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Parameters</div>
                <div className="text-lg font-bold">
                  {formatNumber(bestArchitecture.parameters)}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">FLOPs</div>
                <div className="text-lg font-bold">
                  {formatNumber(bestArchitecture.flops)}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Iteration</div>
                <div className="text-lg font-bold">
                  {bestArchitecture.searchIteration}
                </div>
              </div>
            </div>
            
            <div>
              <div className="text-sm text-muted-foreground mb-2">Architecture Layers</div>
              <div className="flex flex-wrap gap-1">
                {bestArchitecture.layers.map((layer, index) => (
                  <Badge key={index} variant="secondary" className="text-xs">
                    {layer}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Architectures */}
      {architectures.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Recent Architectures
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {architectures.slice(-5).reverse().map((arch, index) => (
                <div 
                  key={arch.id} 
                  className={`p-3 rounded-lg border transition-colors ${
                    arch.id === bestArchitecture?.id 
                      ? 'border-green-500/20 bg-green-500/5' 
                      : 'border-muted'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-muted-foreground" />
                        <span className="font-medium">{arch.name}</span>
                        {arch.id === bestArchitecture?.id && (
                          <Badge variant="secondary" className="text-xs">
                            Best
                          </Badge>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div className="text-center">
                        <div className="font-medium text-green-600">
                          {(arch.accuracy * 100).toFixed(2)}%
                        </div>
                        <div className="text-xs text-muted-foreground">Accuracy</div>
                      </div>
                      <div className="text-center">
                        <div className="font-medium">
                          {formatNumber(arch.parameters)}
                        </div>
                        <div className="text-xs text-muted-foreground">Params</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {isSearching && (
        <Card className="border-blue-500/20 bg-blue-500/5">
          <CardContent className="py-4">
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 rounded-full bg-blue-500 animate-pulse"></div>
              <span className="text-sm text-blue-600 font-medium">
                Architecture search is running... (Iteration {currentIteration})
              </span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};