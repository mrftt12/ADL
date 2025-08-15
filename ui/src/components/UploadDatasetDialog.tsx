import { useState, useCallback } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { apiClient, formatFileSize } from "@/lib/api-client";
import { Upload, File, Loader2, X, Database } from "lucide-react";

interface UploadDatasetDialogProps {
  onDatasetUploaded?: () => void;
}

export const UploadDatasetDialog = ({ onDatasetUploaded }: UploadDatasetDialogProps) => {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    targetColumn: "",
    dataType: ""
  });
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      // Validate file type
      const allowedTypes = ['.csv', '.json', '.txt', '.tsv'];
      const fileExtension = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
      
      if (!allowedTypes.includes(fileExtension)) {
        toast({
          title: "Invalid File Type",
          description: "Please upload a CSV, JSON, TXT, or TSV file",
          variant: "destructive"
        });
        return;
      }

      // Validate file size (max 50MB)
      if (selectedFile.size > 50 * 1024 * 1024) {
        toast({
          title: "File Too Large",
          description: "Please upload a file smaller than 50MB",
          variant: "destructive"
        });
        return;
      }

      setFile(selectedFile);
      
      // Auto-fill name if empty
      if (!formData.name) {
        const nameWithoutExtension = selectedFile.name.replace(/\.[^/.]+$/, "");
        setFormData(prev => ({ ...prev, name: nameWithoutExtension }));
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file || !formData.name || !formData.dataType) {
      toast({
        title: "Validation Error",
        description: "Please fill in all required fields and select a file",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    setUploading(true);

    try {
      // Upload file using the API client
      const response = await apiClient.uploadDataset(
        file,
        formData.name,
        formData.description
      );

      if (response.error) {
        throw new Error(response.error);
      }

      toast({
        title: "Dataset Uploaded",
        description: `${formData.name} has been uploaded successfully`,
      });

      // Reset form
      setFormData({
        name: "",
        description: "",
        targetColumn: "",
        dataType: ""
      });
      setFile(null);
      setOpen(false);
      onDatasetUploaded?.();

    } catch (error: any) {
      console.error('Error uploading dataset:', error);
      toast({
        title: "Upload Failed",
        description: error.message || "Failed to upload dataset",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
      setUploading(false);
    }
  };

  const removeFile = () => {
    setFile(null);
  };

  const resetForm = () => {
    setFormData({
      name: "",
      description: "",
      targetColumn: "",
      dataType: ""
    });
    setFile(null);
  };

  // formatFileSize is now imported from api-client

  return (
    <Dialog open={open} onOpenChange={(newOpen) => {
      setOpen(newOpen);
      if (!newOpen) resetForm();
    }}>
      <DialogTrigger asChild>
        <Button variant="gradient">
          <Upload className="w-4 h-4" />
          Upload Dataset
        </Button>
      </DialogTrigger>
      
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold">Upload Dataset</DialogTitle>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* File Upload Area */}
          <div className="space-y-2">
            <Label>Dataset File *</Label>
            {!file ? (
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-muted-foreground/50 transition-colors">
                <Database className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Drag and drop your dataset file here, or
                  </p>
                  <Label htmlFor="file-upload" className="cursor-pointer">
                    <Button type="button" variant="outline" asChild>
                      <span>Choose File</span>
                    </Button>
                  </Label>
                  <Input
                    id="file-upload"
                    type="file"
                    accept=".csv,.json,.txt,.tsv"
                    onChange={handleFileChange}
                    className="hidden"
                    disabled={loading}
                  />
                  <p className="text-xs text-muted-foreground">
                    Supports CSV, JSON, TXT, TSV files up to 50MB
                  </p>
                </div>
              </div>
            ) : (
              <div className="border rounded-lg p-4 bg-muted/5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <File className="w-8 h-8 text-primary" />
                    <div>
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={removeFile}
                    disabled={loading}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Dataset Name *</Label>
              <Input
                id="name"
                placeholder="e.g., Customer Data"
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="dataType">Data Type *</Label>
              <Select 
                value={formData.dataType} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, dataType: value }))}
                disabled={loading}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select data type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="tabular">Tabular Data</SelectItem>
                  <SelectItem value="text">Text Data</SelectItem>
                  <SelectItem value="time_series">Time Series</SelectItem>
                  <SelectItem value="image_paths">Image Paths</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              placeholder="Describe your dataset, its source, and intended use..."
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              disabled={loading}
              rows={3}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="targetColumn">Target Column</Label>
            <Input
              id="targetColumn"
              placeholder="e.g., label, target, y (for supervised learning)"
              value={formData.targetColumn}
              onChange={(e) => setFormData(prev => ({ ...prev, targetColumn: e.target.value }))}
              disabled={loading}
            />
            <p className="text-xs text-muted-foreground">
              Specify the column name that contains the target variable (optional)
            </p>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => setOpen(false)}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="gradient"
              disabled={loading || !file}
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Upload Dataset
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};