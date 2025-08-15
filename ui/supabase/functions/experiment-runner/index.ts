import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.53.0'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface Database {
  public: {
    Tables: {
      experiments: {
        Row: {
          id: string
          project_id: string
          name: string
          status: 'queued' | 'running' | 'completed' | 'failed'
          progress: number | null
          accuracy: number | null
          loss: number | null
          runtime_seconds: number | null
          started_at: string | null
          completed_at: string | null
          created_at: string
          updated_at: string
          architecture_config: any
          hyperparameters: any
          metrics: any
        }
        Insert: {
          id?: string
          project_id: string
          name: string
          status?: 'queued' | 'running' | 'completed' | 'failed'
          progress?: number | null
          accuracy?: number | null
          loss?: number | null
          runtime_seconds?: number | null
          started_at?: string | null
          completed_at?: string | null
          created_at?: string
          updated_at?: string
          architecture_config?: any
          hyperparameters?: any
          metrics?: any
        }
        Update: {
          id?: string
          project_id?: string
          name?: string
          status?: 'queued' | 'running' | 'completed' | 'failed'
          progress?: number | null
          accuracy?: number | null
          loss?: number | null
          runtime_seconds?: number | null
          started_at?: string | null
          completed_at?: string | null
          created_at?: string
          updated_at?: string
          architecture_config?: any
          hyperparameters?: any
          metrics?: any
        }
      }
    }
  }
}

Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient<Database>(
      'https://ktgzaciforpkhnfdtbqm.supabase.co',
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt0Z3phY2lmb3Jwa2huZmR0YnFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUxMzk2NjYsImV4cCI6MjA2MDcxNTY2Nn0.o0uZDsoRWNTLHL2qOxWSNI-rHDSMW7ixPc1aJwYo28k'
    )

    // Get a queued experiment
    const { data: queuedExperiments, error: fetchError } = await supabase
      .from('experiments')
      .select('*')
      .eq('status', 'queued')
      .order('created_at', { ascending: true })
      .limit(1)

    console.log('Queued experiments query result:', { queuedExperiments, fetchError })

    if (fetchError) {
      console.error('Error fetching queued experiments:', fetchError)
      return new Response(
        JSON.stringify({ error: 'Failed to fetch experiments' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
      )
    }

    if (!queuedExperiments || queuedExperiments.length === 0) {
      return new Response(
        JSON.stringify({ message: 'No queued experiments found' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const experiment = queuedExperiments[0]
    console.log(`Starting experiment: ${experiment.name} (${experiment.id})`)

    // Update experiment to running status
    const { error: updateError } = await supabase
      .from('experiments')
      .update({
        status: 'running',
        started_at: new Date().toISOString(),
        progress: 0,
        updated_at: new Date().toISOString()
      })
      .eq('id', experiment.id)

    if (updateError) {
      console.error('Error updating experiment status:', updateError)
      return new Response(
        JSON.stringify({ error: 'Failed to update experiment status' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
      )
    }

    // Start background simulation of the experiment
    EdgeRuntime.waitUntil(simulateExperiment(supabase, experiment.id))

    return new Response(
      JSON.stringify({ 
        message: `Experiment ${experiment.name} started successfully`,
        experimentId: experiment.id 
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Unexpected error:', error)
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
    )
  }
})

async function simulateExperiment(supabase: any, experimentId: string) {
  console.log(`Simulating experiment ${experimentId}`)
  
  try {
    // Simulate AutoML training with progress updates
    const totalDuration = 30000 // 30 seconds for demo
    const updateInterval = 2000 // Update every 2 seconds
    const totalSteps = totalDuration / updateInterval
    
    for (let step = 1; step <= totalSteps; step++) {
      await new Promise(resolve => setTimeout(resolve, updateInterval))
      
      const progress = Math.floor((step / totalSteps) * 100)
      const currentAccuracy = 0.5 + (progress / 100) * 0.4 + (Math.random() * 0.1 - 0.05) // 50-90% with noise
      const currentLoss = 2.0 - (progress / 100) * 1.5 + (Math.random() * 0.2 - 0.1) // 2.0 to 0.5 with noise
      
      console.log(`Experiment ${experimentId} - Step ${step}/${totalSteps} - Progress: ${progress}%`)
      
      await supabase
        .from('experiments')
        .update({
          progress,
          accuracy: Number(currentAccuracy.toFixed(4)),
          loss: Number(currentLoss.toFixed(4)),
          runtime_seconds: Math.floor((step * updateInterval) / 1000),
          updated_at: new Date().toISOString(),
          metrics: {
            training_accuracy: currentAccuracy,
            validation_accuracy: currentAccuracy * 0.95,
            training_loss: currentLoss,
            validation_loss: currentLoss * 1.05,
            epoch: step,
            learning_rate: 0.001 * Math.pow(0.95, step)
          }
        })
        .eq('id', experimentId)
    }
    
    // Complete the experiment
    const finalAccuracy = 0.85 + Math.random() * 0.1 // 85-95%
    const finalLoss = 0.3 + Math.random() * 0.2 // 0.3-0.5
    
    await supabase
      .from('experiments')
      .update({
        status: 'completed',
        progress: 100,
        accuracy: Number(finalAccuracy.toFixed(4)),
        loss: Number(finalLoss.toFixed(4)),
        runtime_seconds: totalDuration / 1000,
        completed_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        architecture_config: {
          model_type: 'neural_network',
          layers: [
            { type: 'dense', units: 128, activation: 'relu' },
            { type: 'dropout', rate: 0.2 },
            { type: 'dense', units: 64, activation: 'relu' },
            { type: 'dense', units: 1, activation: 'sigmoid' }
          ],
          optimizer: 'adam',
          loss_function: 'binary_crossentropy'
        },
        hyperparameters: {
          learning_rate: 0.001,
          batch_size: 32,
          epochs: 50,
          optimizer: 'adam',
          dropout_rate: 0.2
        }
      })
      .eq('id', experimentId)
    
    console.log(`Experiment ${experimentId} completed successfully`)
    
  } catch (error) {
    console.error(`Error simulating experiment ${experimentId}:`, error)
    
    // Mark experiment as failed
    await supabase
      .from('experiments')
      .update({
        status: 'failed',
        updated_at: new Date().toISOString()
      })
      .eq('id', experimentId)
  }
}