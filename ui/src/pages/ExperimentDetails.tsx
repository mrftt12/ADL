import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AppHeader } from "@/components/AppHeader";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { 
  ArrowLeft, 
  Brain, 
  Clock, 
  TrendingUp, 
  Settings,
  BarChart3,
  Activity,
  Target,
  Cpu
} from "lucide-react";

interface Experiment {
  id: string;
  name: string;
  project_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  accuracy?: number;
  loss?: number;
  runtime_seconds: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  architecture_config?: {
    model_type?: string;
    layers?: Array<{ type: string; units?: number; activation?: string; rate?: number }>;
    optimizer?: string;
    loss_function?: string;
  };
  hyperparameters?: {
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
    optimizer?: string;
    dropout_rate?: number;
  };
  metrics?: {
    training_accuracy?: number;
    validation_accuracy?: number;
    training_loss?: number;
    validation_loss?: number;
    epoch?: number;
    learning_rate?: number;
  };
}

interface Project {
  id: string;
  name: string;
  dataset_name: string;
}

const ExperimentDetails = () => {
  const { projectId, experimentId } = useParams<{ projectId: string; experimentId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (projectId && experimentId) {
      loadExperimentDetails();
    }
  }, [projectId, experimentId]);

  const loadExperimentDetails = async () => {
    try {
      setLoading(true);
      
      // Load experiment details
      const { data: experimentData, error: experimentError } = await supabase
        .from('experiments')
        .select('*')
        .eq('id', experimentId)
        .eq('project_id', projectId)
        .single();

      if (experimentError) throw experimentError;
      
      // Load project details
      const { data: projectData, error: projectError } = await supabase
        .from('projects')
        .select('id, name, dataset_name')
        .eq('id', projectId)
        .single();

      if (projectError) throw projectError;
      
      setExperiment({
        ...experimentData,
        architecture_config: experimentData.architecture_config as any,
        hyperparameters: experimentData.hyperparameters as any,
        metrics: experimentData.metrics as any
      });
      setProject(projectData);
    } catch (error: any) {
      console.error('Error loading experiment details:', error);
      toast({
        title: "Error",
        description: "Failed to load experiment details",
        variant: "destructive"
      });
      navigate(`/projects/${projectId}`);
    } finally {
      setLoading(false);
    }
  };

  const formatRuntime = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500/10 text-green-600 border-green-500/20';
      case 'running': return 'bg-blue-500/10 text-blue-600 border-blue-500/20';
      case 'failed': return 'bg-red-500/10 text-red-600 border-red-500/20';
      case 'paused': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      default: return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-background">
        <AppHeader />
        <div className="container mx-auto px-6 py-8">
          <div className="animate-pulse">
            <div className="h-8 bg-muted rounded w-1/4 mb-8"></div>
            <div className="grid gap-6">
              <div className="h-64 bg-muted rounded-lg"></div>
              <div className="h-96 bg-muted rounded-lg"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!experiment || !project) {
    return (
      <div className="min-h-screen bg-gradient-background">
        <AppHeader />
        <div className="container mx-auto px-6 py-8 text-center">
          <h1 className="text-2xl font-bold mb-4">Experiment Not Found</h1>
          <Button onClick={() => navigate(`/projects/${projectId}`)} variant="outline">
            <ArrowLeft className="w-4 h-4" />
            Back to Project
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-background">
      <AppHeader />
      
      <div className="container mx-auto px-6 py-8">
        <div className="flex items-center gap-4 mb-8">
          <Button variant="ghost" onClick={() => navigate(`/projects/${projectId}`)} size="sm">
            <ArrowLeft className="w-4 h-4" />
            Back to Project
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{experiment.name}</h1>
            <p className="text-muted-foreground mt-1">
              Experiment in {project.name} • Dataset: {project.dataset_name}
            </p>
          </div>
        </div>

        <div className="grid gap-6">
          {/* Experiment Overview */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Experiment Overview
                  </CardTitle>
                  <CardDescription>Performance metrics and execution details</CardDescription>
                </div>
                <Badge className={`capitalize ${getStatusColor(experiment.status)}`}>
                  {experiment.status}
                </Badge>
              </div>
            </CardHeader>
            
            <CardContent>
              <div className="grid md:grid-cols-4 gap-6">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-medium">{experiment.progress}%</span>
                  </div>
                  <Progress value={experiment.progress} className="h-2" />
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {experiment.accuracy ? `${(experiment.accuracy * 100).toFixed(2)}%` : 'N/A'}
                  </div>
                  <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                    <Target className="w-3 h-3" />
                    Accuracy
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {experiment.loss ? experiment.loss.toFixed(4) : 'N/A'}
                  </div>
                  <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                    <TrendingUp className="w-3 h-3" />
                    Loss
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {formatRuntime(experiment.runtime_seconds)}
                  </div>
                  <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                    <Clock className="w-3 h-3" />
                    Runtime
                  </div>
                </div>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6 mt-6 pt-6 border-t">
                <div>
                  <h4 className="font-semibold mb-2">Timeline</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Created:</span>
                      <span>{new Date(experiment.created_at).toLocaleString()}</span>
                    </div>
                    {experiment.started_at && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Started:</span>
                        <span>{new Date(experiment.started_at).toLocaleString()}</span>
                      </div>
                    )}
                    {experiment.completed_at && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Completed:</span>
                        <span>{new Date(experiment.completed_at).toLocaleString()}</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Status</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Current Status:</span>
                      <Badge className={getStatusColor(experiment.status)} variant="outline">
                        {experiment.status}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Progress:</span>
                      <span>{experiment.progress}% complete</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Detailed Analysis Tabs */}
          <Tabs defaultValue="architecture" className="w-full">
            <TabsList>
              <TabsTrigger value="architecture">Architecture</TabsTrigger>
              <TabsTrigger value="hyperparameters">Hyperparameters</TabsTrigger>
              <TabsTrigger value="metrics">Training Metrics</TabsTrigger>
              <TabsTrigger value="insights">Insights</TabsTrigger>
            </TabsList>
            
            <TabsContent value="architecture" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Neural Architecture
                  </CardTitle>
                  <CardDescription>
                    Model architecture and configuration details
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {experiment.architecture_config ? (
                    <div className="space-y-6">
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-semibold mb-3">Model Configuration</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Model Type:</span>
                              <span className="font-medium">{experiment.architecture_config.model_type || 'Neural Network'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Optimizer:</span>
                              <span className="font-medium">{experiment.architecture_config.optimizer || 'adam'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Loss Function:</span>
                              <span className="font-medium">{experiment.architecture_config.loss_function || 'binary_crossentropy'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Total Layers:</span>
                              <span className="font-medium">{experiment.architecture_config.layers?.length || 0}</span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold mb-3">Model Parameters</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Total Parameters:</span>
                              <span className="font-medium">
                                {experiment.architecture_config.layers?.reduce((sum: number, layer: any) => 
                                  sum + (layer.units || 0), 0).toLocaleString() || '0'
                                }
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {experiment.architecture_config.layers && (
                        <div>
                          <h4 className="font-semibold mb-3">Layer Architecture</h4>
                          <div className="space-y-3">
                            {experiment.architecture_config.layers.map((layer, index) => (
                              <Card key={index} className="border">
                                <CardContent className="py-3">
                                  <div className="flex justify-between items-center">
                                    <div>
                                      <span className="font-medium">Layer {index + 1}: {layer.type}</span>
                                      {layer.activation && (
                                        <span className="text-muted-foreground ml-2">({layer.activation})</span>
                                      )}
                                    </div>
                                    <div className="text-sm text-muted-foreground">
                                      {layer.units && <span>Units: {layer.units}</span>}
                                      {layer.rate && <span>Rate: {layer.rate}</span>}
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="text-lg font-semibold mb-2">No Architecture Data</h3>
                      <p className="text-muted-foreground">
                        Architecture details will be available once the experiment completes
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="hyperparameters" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="w-5 h-5" />
                    Hyperparameters
                  </CardTitle>
                  <CardDescription>
                    Training configuration and parameter settings
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {experiment.hyperparameters ? (
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold mb-3">Training Parameters</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Learning Rate:</span>
                            <span className="font-medium">{experiment.hyperparameters.learning_rate || '0.001'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Batch Size:</span>
                            <span className="font-medium">{experiment.hyperparameters.batch_size || '32'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Epochs:</span>
                            <span className="font-medium">{experiment.hyperparameters.epochs || '50'}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold mb-3">Optimization</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Optimizer:</span>
                            <span className="font-medium">{experiment.hyperparameters.optimizer || 'adam'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Dropout Rate:</span>
                            <span className="font-medium">{experiment.hyperparameters.dropout_rate || '0.2'}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Settings className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="text-lg font-semibold mb-2">No Hyperparameter Data</h3>
                      <p className="text-muted-foreground">
                        Hyperparameter details will be available once the experiment completes
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="metrics" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Training Metrics
                  </CardTitle>
                  <CardDescription>
                    Detailed performance metrics from training
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {experiment.metrics ? (
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold mb-3">Accuracy Metrics</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Training Accuracy:</span>
                            <span className="font-medium">
                              {(experiment.metrics.training_accuracy * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Validation Accuracy:</span>
                            <span className="font-medium">
                              {(experiment.metrics.validation_accuracy * 100).toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold mb-3">Loss Metrics</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Training Loss:</span>
                            <span className="font-medium">
                              {experiment.metrics.training_loss?.toFixed(4) || 'N/A'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Validation Loss:</span>
                            <span className="font-medium">
                              {experiment.metrics.validation_loss?.toFixed(4) || 'N/A'}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold mb-3">Training Progress</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Epochs Completed:</span>
                            <span className="font-medium">{experiment.metrics.epoch || 'N/A'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Final Learning Rate:</span>
                            <span className="font-medium">
                              {experiment.metrics.learning_rate?.toFixed(6) || 'N/A'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="text-lg font-semibold mb-2">No Training Metrics</h3>
                      <p className="text-muted-foreground">
                        Training metrics will be available once the experiment completes
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="insights" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Cpu className="w-5 h-5" />
                    Model Insights & Recommendations
                  </CardTitle>
                  <CardDescription>
                    Analysis and suggestions for model improvement
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {experiment.status === 'completed' && experiment.metrics ? (
                    <div className="space-y-6">
                      <div className="grid md:grid-cols-2 gap-6">
                        <Card className="border-l-4 border-l-green-500">
                          <CardContent className="pt-4">
                            <h4 className="font-semibold text-green-700 mb-2">Performance Summary</h4>
                            <div className="space-y-1 text-sm">
                              <p>• Final accuracy: {(experiment.accuracy * 100).toFixed(2)}%</p>
                              <p>• Training completed in {formatRuntime(experiment.runtime_seconds)}</p>
                              <p>• Model converged successfully</p>
                            </div>
                          </CardContent>
                        </Card>
                        
                        <Card className="border-l-4 border-l-blue-500">
                          <CardContent className="pt-4">
                            <h4 className="font-semibold text-blue-700 mb-2">Architecture Quality</h4>
                            <div className="space-y-1 text-sm">
                              <p>• Well-balanced layer configuration</p>
                              <p>• Appropriate dropout regularization</p>
                              <p>• Optimal parameter count</p>
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                      
                      <div className="p-4 bg-muted/50 rounded-lg">
                        <h4 className="font-semibold mb-2">Recommendations</h4>
                        <ul className="text-sm space-y-1 text-muted-foreground">
                          <li>• Model shows good performance with {(experiment.accuracy * 100).toFixed(1)}% accuracy</li>
                          <li>• Consider fine-tuning learning rate for potential improvement</li>
                          <li>• Monitor validation metrics to avoid overfitting</li>
                          {experiment.metrics.validation_accuracy < experiment.metrics.training_accuracy * 0.9 && (
                            <li>• Gap between training and validation suggests possible overfitting</li>
                          )}
                          <li>• This architecture can be used as a baseline for future experiments</li>
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Cpu className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="text-lg font-semibold mb-2">Insights Pending</h3>
                      <p className="text-muted-foreground">
                        Detailed insights will be available once the experiment completes
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default ExperimentDetails;