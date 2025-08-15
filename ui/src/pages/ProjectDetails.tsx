import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AppHeader } from "@/components/AppHeader";
import { CreateExperimentDialog } from "@/components/CreateExperimentDialog";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { 
  ArrowLeft, 
  Brain, 
  Clock, 
  Database, 
  TrendingUp, 
  Play, 
  Pause, 
  Settings, 
  Download,
  BarChart3,
  Loader2,
  Eye
} from "lucide-react";

interface Project {
  id: string;
  name: string;
  description?: string;
  status: 'draft' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  best_accuracy?: number;
  runtime_minutes: number;
  dataset_name: string;
  dataset_size?: number;
  model_type: 'classification' | 'regression' | 'object_detection' | 'nlp' | 'time_series';
  created_at: string;
  updated_at: string;
}

interface Experiment {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  accuracy?: number;
  loss?: number;
  runtime_seconds: number;
  created_at: string;
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

const ProjectDetails = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [project, setProject] = useState<Project | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) {
      loadProjectDetails();
    }
  }, [id]);

  const loadProjectDetails = async () => {
    try {
      setLoading(true);
      
      // Load project details
      const { data: projectData, error: projectError } = await supabase
        .from('projects')
        .select('*')
        .eq('id', id)
        .single();

      if (projectError) throw projectError;
      
      // Load experiments for this project
      const { data: experimentsData, error: experimentsError } = await supabase
        .from('experiments')
        .select('*')
        .eq('project_id', id)
        .order('created_at', { ascending: false });

      if (experimentsError) throw experimentsError;
      
      setProject(projectData);
      setExperiments((experimentsData || []).map(exp => ({
        ...exp,
        architecture_config: exp.architecture_config as any,
        hyperparameters: exp.hyperparameters as any,
        metrics: exp.metrics as any
      })));
    } catch (error: any) {
      console.error('Error loading project details:', error);
      toast({
        title: "Error",
        description: "Failed to load project details",
        variant: "destructive"
      });
      navigate("/");
    } finally {
      setLoading(false);
    }
  };

  const handleProjectAction = async (action: 'pause' | 'resume') => {
    if (!project) return;
    
    try {
      const newStatus = action === 'pause' ? 'paused' : 'running';
      
      const { error } = await supabase
        .from('projects')
        .update({ status: newStatus })
        .eq('id', project.id);

      if (error) throw error;

      setProject(prev => prev ? { ...prev, status: newStatus as any } : null);
      
      toast({
        title: "Success",
        description: `Project ${action}d successfully`,
      });
    } catch (error: any) {
      console.error(`Error ${action}ing project:`, error);
      toast({
        title: "Error",
        description: `Failed to ${action} project`,
        variant: "destructive"
      });
    }
  };

  const formatModelType = (modelType: string) => {
    return modelType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const formatRuntime = (minutes: number) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
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

  if (!project) {
    return (
      <div className="min-h-screen bg-gradient-background">
        <AppHeader />
        <div className="container mx-auto px-6 py-8 text-center">
          <h1 className="text-2xl font-bold mb-4">Project Not Found</h1>
          <Button onClick={() => navigate("/")} variant="outline">
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
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
          <Button variant="ghost" onClick={() => navigate("/")} size="sm">
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{project.name}</h1>
            {project.description && (
              <p className="text-muted-foreground mt-1">{project.description}</p>
            )}
          </div>
        </div>

        <div className="grid gap-6">
          {/* Project Overview */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Project Overview
                  </CardTitle>
                  <CardDescription>Current status and performance metrics</CardDescription>
                </div>
                <div className="flex items-center gap-3">
                  <Badge className={`capitalize ${getStatusColor(project.status)}`}>
                    {project.status}
                  </Badge>
                  {project.status === 'running' ? (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleProjectAction('pause')}
                    >
                      <Pause className="w-4 h-4" />
                      Pause
                    </Button>
                  ) : project.status === 'paused' ? (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleProjectAction('resume')}
                    >
                      <Play className="w-4 h-4" />
                      Resume
                    </Button>
                  ) : null}
                </div>
              </div>
            </CardHeader>
            
            <CardContent>
              <div className="grid md:grid-cols-4 gap-6">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-medium">{project.progress}%</span>
                  </div>
                  <Progress value={project.progress} className="h-2" />
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {project.best_accuracy ? `${(project.best_accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                  <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                    <TrendingUp className="w-3 h-3" />
                    Best Accuracy
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {formatRuntime(project.runtime_minutes)}
                  </div>
                  <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                    <Clock className="w-3 h-3" />
                    Runtime
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {project.dataset_size?.toLocaleString() || 'N/A'}
                  </div>
                  <div className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                    <Database className="w-3 h-3" />
                    Samples
                  </div>
                </div>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6 mt-6 pt-6 border-t">
                <div>
                  <h4 className="font-semibold mb-2">Dataset Information</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Name:</span>
                      <span>{project.dataset_name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Type:</span>
                      <span>{formatModelType(project.model_type)}</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Timeline</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Created:</span>
                      <span>{new Date(project.created_at).toLocaleDateString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Last Updated:</span>
                      <span>{new Date(project.updated_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Experiments Tab */}
          <Tabs defaultValue="experiments" className="w-full">
            <TabsList>
              <TabsTrigger value="experiments">Experiments ({experiments.length})</TabsTrigger>
              <TabsTrigger value="architecture">Architecture Search</TabsTrigger>
              <TabsTrigger value="hyperparameters">Hyperparameters</TabsTrigger>
              <TabsTrigger value="results">Results & Analysis</TabsTrigger>
            </TabsList>
            
            <TabsContent value="experiments" className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Experiment History</h3>
                <CreateExperimentDialog 
                  projectId={project.id} 
                  onExperimentCreated={loadProjectDetails}
                />
              </div>
              
              {experiments.length === 0 ? (
                <Card className="text-center py-12">
                  <CardContent>
                    <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <h4 className="text-lg font-semibold mb-2">No Experiments Yet</h4>
                    <p className="text-muted-foreground mb-4">
                      Start your first experiment to begin automated model training
                    </p>
                    <CreateExperimentDialog 
                      projectId={project.id} 
                      onExperimentCreated={loadProjectDetails}
                    />
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-4">
                  {experiments.map((experiment) => (
                     <Card key={experiment.id}>
                       <CardContent className="py-4">
                         <div className="flex items-center justify-between">
                           <div className="flex-1">
                             <h4 className="font-semibold">{experiment.name}</h4>
                             <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                               <span>Started {new Date(experiment.created_at).toLocaleDateString()}</span>
                               <span>{Math.floor(experiment.runtime_seconds / 60)}m runtime</span>
                               {experiment.accuracy && <span>{(experiment.accuracy * 100).toFixed(1)}% accuracy</span>}
                             </div>
                           </div>
                           <div className="flex items-center gap-3">
                             <Badge className={getStatusColor(experiment.status)}>
                               {experiment.status}
                             </Badge>
                             <span className="text-sm font-medium">{experiment.progress}%</span>
                             {experiment.status === 'completed' && (
                               <Button
                                 variant="outline"
                                 size="sm"
                                 onClick={() => navigate(`/projects/${project.id}/experiments/${experiment.id}`)}
                               >
                                 <Eye className="w-4 h-4" />
                                 View Details
                               </Button>
                             )}
                           </div>
                         </div>
                       </CardContent>
                     </Card>
                  ))}
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="architecture" className="space-y-4">
              <div className="grid gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="w-5 h-5" />
                      Neural Architecture Search Results
                    </CardTitle>
                    <CardDescription>
                      Discovered architectures ranked by performance
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {experiments.filter(exp => exp.status === 'completed').length === 0 ? (
                        <div className="text-center py-8">
                          <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Architecture Data Yet</h3>
                          <p className="text-muted-foreground">
                            Complete some experiments to see architecture search results
                          </p>
                        </div>
                      ) : (
                        experiments
                          .filter(exp => exp.status === 'completed' && exp.architecture_config)
                          .slice(0, 5)
                          .map((exp, index) => (
                            <Card key={exp.id} className="border-l-4 border-l-primary/20">
                              <CardContent className="pt-4">
                                <div className="flex justify-between items-start mb-3">
                                  <div>
                                    <h4 className="font-semibold">Architecture #{index + 1}</h4>
                                    <p className="text-sm text-muted-foreground">
                                      From experiment: {exp.name}
                                    </p>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-lg font-bold text-primary">
                                      {exp.accuracy ? (exp.accuracy * 100).toFixed(2) : '0.00'}%
                                    </div>
                                    <div className="text-xs text-muted-foreground">Accuracy</div>
                                  </div>
                                </div>
                                
                                {exp.architecture_config && (
                                  <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                      <span className="text-muted-foreground">Model Type:</span>
                                      <span className="font-medium">{exp.architecture_config.model_type || 'Neural Network'}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                      <span className="text-muted-foreground">Layers:</span>
                                      <span className="font-medium">{exp.architecture_config.layers?.length || 0}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                      <span className="text-muted-foreground">Optimizer:</span>
                                      <span className="font-medium">{exp.architecture_config.optimizer || 'adam'}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                      <span className="text-muted-foreground">Parameters:</span>
                                      <span className="font-medium">
                                        {exp.architecture_config.layers?.reduce((sum: number, layer: any) => 
                                          sum + (layer.units || 0), 0).toLocaleString() || '0'
                                        }
                                      </span>
                                    </div>
                                  </div>
                                )}
                              </CardContent>
                            </Card>
                          ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            <TabsContent value="hyperparameters" className="space-y-4">
              <div className="grid gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Settings className="w-5 h-5" />
                      Hyperparameter Optimization
                    </CardTitle>
                    <CardDescription>
                      Best performing hyperparameter configurations
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {experiments.filter(exp => exp.status === 'completed').length === 0 ? (
                        <div className="text-center py-8">
                          <Settings className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2">No Hyperparameter Data Yet</h3>
                          <p className="text-muted-foreground">
                            Complete some experiments to see hyperparameter optimization results
                          </p>
                        </div>
                      ) : (
                        <>
                          {/* Best Configuration */}
                          {(() => {
                            const bestExperiment = experiments
                              .filter(exp => exp.status === 'completed' && exp.accuracy)
                              .sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0))[0];
                            
                            if (!bestExperiment) return null;
                            
                            return (
                              <Card className="border-l-4 border-l-green-500">
                                <CardContent className="pt-4">
                                  <div className="flex justify-between items-center mb-3">
                                    <h4 className="font-semibold text-green-700">Best Configuration</h4>
                                    <Badge variant="secondary" className="bg-green-100 text-green-800">
                                      {(bestExperiment.accuracy * 100).toFixed(2)}% accuracy
                                    </Badge>
                                  </div>
                                  
                                  {bestExperiment.hyperparameters && (
                                    <div className="grid grid-cols-2 gap-4">
                                      <div className="space-y-2">
                                        <div className="flex justify-between text-sm">
                                          <span className="text-muted-foreground">Learning Rate:</span>
                                          <span className="font-medium">{bestExperiment.hyperparameters.learning_rate || '0.001'}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                          <span className="text-muted-foreground">Batch Size:</span>
                                          <span className="font-medium">{bestExperiment.hyperparameters.batch_size || '32'}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                          <span className="text-muted-foreground">Epochs:</span>
                                          <span className="font-medium">{bestExperiment.hyperparameters.epochs || '50'}</span>
                                        </div>
                                      </div>
                                      <div className="space-y-2">
                                        <div className="flex justify-between text-sm">
                                          <span className="text-muted-foreground">Optimizer:</span>
                                          <span className="font-medium">{bestExperiment.hyperparameters.optimizer || 'adam'}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                          <span className="text-muted-foreground">Dropout Rate:</span>
                                          <span className="font-medium">{bestExperiment.hyperparameters.dropout_rate || '0.2'}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                          <span className="text-muted-foreground">Final Loss:</span>
                                          <span className="font-medium">{bestExperiment.loss?.toFixed(4) || 'N/A'}</span>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </CardContent>
                              </Card>
                            );
                          })()}
                          
                          {/* All Configurations */}
                          <Card>
                            <CardHeader>
                              <CardTitle className="text-lg">All Configurations</CardTitle>
                            </CardHeader>
                            <CardContent>
                              <div className="space-y-3">
                                {experiments
                                  .filter(exp => exp.status === 'completed')
                                  .sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0))
                                  .map((exp, index) => (
                                    <Card key={exp.id} className="border">
                                      <CardContent className="py-3">
                                        <div className="flex justify-between items-center">
                                          <div>
                                            <h5 className="font-medium">{exp.name}</h5>
                                            <div className="flex gap-4 text-xs text-muted-foreground mt-1">
                                              <span>LR: {exp.hyperparameters?.learning_rate || '0.001'}</span>
                                              <span>Batch: {exp.hyperparameters?.batch_size || '32'}</span>
                                              <span>Dropout: {exp.hyperparameters?.dropout_rate || '0.2'}</span>
                                            </div>
                                          </div>
                                          <div className="text-right">
                                            <div className="font-bold">
                                              {exp.accuracy ? (exp.accuracy * 100).toFixed(2) : '0.00'}%
                                            </div>
                                            <div className="text-xs text-muted-foreground">
                                              Loss: {exp.loss?.toFixed(4) || 'N/A'}
                                            </div>
                                          </div>
                                        </div>
                                      </CardContent>
                                    </Card>
                                  ))}
                              </div>
                            </CardContent>
                          </Card>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            <TabsContent value="results" className="space-y-4">
              <div className="grid gap-4">
                {/* Performance Overview */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5" />
                      Performance Overview
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {experiments.filter(exp => exp.status === 'completed').length === 0 ? (
                      <div className="text-center py-8">
                        <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                        <h3 className="text-lg font-semibold mb-2">No Results Yet</h3>
                        <p className="text-muted-foreground">
                          Complete some experiments to see detailed analysis
                        </p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {(() => {
                          const completedExperiments = experiments.filter(exp => exp.status === 'completed');
                          const bestAccuracy = Math.max(...completedExperiments.map(exp => exp.accuracy || 0));
                          const avgAccuracy = completedExperiments.reduce((sum, exp) => sum + (exp.accuracy || 0), 0) / completedExperiments.length;
                          const totalRuntime = completedExperiments.reduce((sum, exp) => sum + (exp.runtime_seconds || 0), 0);
                          const bestLoss = Math.min(...completedExperiments.filter(exp => exp.loss).map(exp => exp.loss || Infinity));
                          
                          return (
                            <>
                              <div className="text-center p-4 bg-primary/5 rounded-lg">
                                <div className="text-2xl font-bold text-primary">{(bestAccuracy * 100).toFixed(1)}%</div>
                                <div className="text-sm text-muted-foreground">Best Accuracy</div>
                              </div>
                              <div className="text-center p-4 bg-secondary/5 rounded-lg">
                                <div className="text-2xl font-bold">{(avgAccuracy * 100).toFixed(1)}%</div>
                                <div className="text-sm text-muted-foreground">Avg Accuracy</div>
                              </div>
                              <div className="text-center p-4 bg-accent/5 rounded-lg">
                                <div className="text-2xl font-bold">{Math.floor(totalRuntime / 60)}m</div>
                                <div className="text-sm text-muted-foreground">Total Runtime</div>
                              </div>
                              <div className="text-center p-4 bg-muted/5 rounded-lg">
                                <div className="text-2xl font-bold">{isFinite(bestLoss) ? bestLoss.toFixed(3) : 'N/A'}</div>
                                <div className="text-sm text-muted-foreground">Best Loss</div>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Experiment Comparison */}
                {experiments.filter(exp => exp.status === 'completed').length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Experiment Comparison</CardTitle>
                      <CardDescription>
                        Compare performance across all completed experiments
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {experiments
                          .filter(exp => exp.status === 'completed')
                          .sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0))
                          .map((exp, index) => (
                            <div key={exp.id} className="flex items-center gap-4 p-3 border rounded-lg">
                              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-bold">
                                {index + 1}
                              </div>
                              <div className="flex-1">
                                <div className="font-medium">{exp.name}</div>
                                <div className="text-sm text-muted-foreground">
                                  {formatRuntime(exp.runtime_seconds || 0)} runtime
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="font-bold">
                                  {exp.accuracy ? (exp.accuracy * 100).toFixed(2) : '0.00'}%
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  Loss: {exp.loss?.toFixed(4) || 'N/A'}
                                </div>
                              </div>
                              <div className="w-24">
                                <Progress 
                                  value={exp.accuracy ? exp.accuracy * 100 : 0} 
                                  className="h-2"
                                />
                              </div>
                            </div>
                          ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Model Insights */}
                {experiments.filter(exp => exp.status === 'completed' && exp.metrics).length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Model Insights</CardTitle>
                      <CardDescription>
                        Key insights from your best performing model
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {(() => {
                        const bestExperiment = experiments
                          .filter(exp => exp.status === 'completed' && exp.accuracy)
                          .sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0))[0];
                        
                        if (!bestExperiment || !bestExperiment.metrics) {
                          return <p className="text-muted-foreground">No detailed metrics available yet.</p>;
                        }
                        
                        return (
                          <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <h4 className="font-semibold mb-2">Training Metrics</h4>
                                <div className="space-y-2 text-sm">
                                  <div className="flex justify-between">
                                    <span>Training Accuracy:</span>
                                    <span className="font-medium">
                                      {(bestExperiment.metrics.training_accuracy * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Validation Accuracy:</span>
                                    <span className="font-medium">
                                      {(bestExperiment.metrics.validation_accuracy * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Final Learning Rate:</span>
                                    <span className="font-medium">
                                      {bestExperiment.metrics.learning_rate?.toFixed(6) || 'N/A'}
                                    </span>
                                  </div>
                                </div>
                              </div>
                              <div>
                                <h4 className="font-semibold mb-2">Loss Metrics</h4>
                                <div className="space-y-2 text-sm">
                                  <div className="flex justify-between">
                                    <span>Training Loss:</span>
                                    <span className="font-medium">
                                      {bestExperiment.metrics.training_loss?.toFixed(4) || 'N/A'}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Validation Loss:</span>
                                    <span className="font-medium">
                                      {bestExperiment.metrics.validation_loss?.toFixed(4) || 'N/A'}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Epochs Completed:</span>
                                    <span className="font-medium">
                                      {bestExperiment.metrics.epoch || 'N/A'}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            <div className="mt-4 p-4 bg-muted/50 rounded-lg">
                              <h4 className="font-semibold mb-2">Recommendations</h4>
                              <ul className="text-sm space-y-1 text-muted-foreground">
                                <li>• Model shows good performance with {(bestExperiment.accuracy * 100).toFixed(1)}% accuracy</li>
                                <li>• Consider fine-tuning learning rate for potential improvement</li>
                                <li>• Monitor validation metrics to avoid overfitting</li>
                                {bestExperiment.metrics.validation_accuracy < bestExperiment.metrics.training_accuracy * 0.9 && (
                                  <li>• Gap between training and validation suggests possible overfitting</li>
                                )}
                              </ul>
                            </div>
                          </div>
                        );
                      })()}
                    </CardContent>
                  </Card>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default ProjectDetails;