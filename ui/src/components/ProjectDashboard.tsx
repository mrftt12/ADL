import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { CreateProjectDialog } from "@/components/CreateProjectDialog";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate } from "react-router-dom";
import { Brain, Clock, Database, TrendingUp, Play, Pause, Eye, RefreshCw } from "lucide-react";

interface Project {
  id: string;
  name: string;
  status: 'draft' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  best_accuracy?: number;
  runtime_minutes: number;
  dataset_name: string;
  model_type: 'classification' | 'regression' | 'object_detection' | 'nlp' | 'time_series';
  created_at: string;
}

export const ProjectDashboard = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const loadProjects = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user) {
        setProjects([]);
        setLoading(false);
        return;
      }

      const { data, error } = await supabase
        .from('projects')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      
      setProjects(data || []);
    } catch (error: any) {
      console.error('Error loading projects:', error);
      toast({
        title: "Error",
        description: "Failed to load projects",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadProjects();
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500/10 text-green-600 border-green-500/20';
      case 'running': return 'bg-blue-500/10 text-blue-600 border-blue-500/20';
      case 'failed': return 'bg-red-500/10 text-red-600 border-red-500/20';
      case 'paused': return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      default: return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
    }
  };

  const getStatusVariant = (status: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (status) {
      case 'completed': return 'default';
      case 'running': return 'default';
      case 'failed': return 'destructive';
      case 'paused': return 'secondary';
      default: return 'outline';
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

  const handleProjectAction = async (projectId: string, action: 'pause' | 'resume') => {
    try {
      const newStatus = action === 'pause' ? 'paused' : 'running';
      
      const { error } = await supabase
        .from('projects')
        .update({ status: newStatus })
        .eq('id', projectId);

      if (error) throw error;

      toast({
        title: "Success",
        description: `Project ${action}d successfully`,
      });

      await loadProjects();
    } catch (error: any) {
      console.error(`Error ${action}ing project:`, error);
      toast({
        title: "Error",
        description: `Failed to ${action} project`,
        variant: "destructive"
      });
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-6 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-muted rounded w-1/4 mb-2"></div>
          <div className="h-4 bg-muted rounded w-1/2 mb-8"></div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-64 bg-muted rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Your AutoML Projects</h2>
          <p className="text-muted-foreground">
            Manage and monitor your automated machine learning experiments
          </p>
        </div>
        <div className="flex gap-3">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <CreateProjectDialog onProjectCreated={loadProjects} />
        </div>
      </div>
      
      {projects.length === 0 ? (
        <Card className="text-center py-12">
          <CardContent>
            <div className="mx-auto w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mb-6">
              <Brain className="w-12 h-12 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">No Projects Yet</h3>
            <p className="text-muted-foreground mb-6">
              Create your first AutoML project to get started with automated deep learning
            </p>
            <CreateProjectDialog onProjectCreated={loadProjects} />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <Card key={project.id} className="hover:shadow-lg transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-lg font-semibold line-clamp-1">
                      {project.name}
                    </CardTitle>
                    <CardDescription className="flex items-center gap-2 mt-1">
                      <Database className="w-4 h-4" />
                      {project.dataset_name}
                    </CardDescription>
                  </div>
                  <Badge 
                    variant={getStatusVariant(project.status)}
                    className={`capitalize ${getStatusColor(project.status)}`}
                  >
                    {project.status}
                  </Badge>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-medium">{project.progress}%</span>
                  </div>
                  <Progress value={project.progress} className="h-2" />
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <TrendingUp className="w-3 h-3" />
                      Accuracy
                    </div>
                    <div className="font-medium">
                      {project.best_accuracy ? `${(project.best_accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      Runtime
                    </div>
                    <div className="font-medium">{formatRuntime(project.runtime_minutes)}</div>
                  </div>
                </div>
                
                <div className="text-sm">
                  <div className="text-muted-foreground">Model Type</div>
                  <div className="font-medium">{formatModelType(project.model_type)}</div>
                </div>
                
                <div className="text-xs text-muted-foreground">
                  Created {new Date(project.created_at).toLocaleDateString()}
                </div>
                
                <div className="flex gap-2 pt-2">
                  {project.status === 'running' ? (
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleProjectAction(project.id, 'pause')}
                    >
                      <Pause className="w-3 h-3" />
                      Pause
                    </Button>
                  ) : project.status === 'paused' ? (
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleProjectAction(project.id, 'resume')}
                    >
                      <Play className="w-3 h-3" />
                      Resume
                    </Button>
                  ) : null}
                  
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => navigate(`/projects/${project.id}`)}
                  >
                    <Eye className="w-3 h-3" />
                    View Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};