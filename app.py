from flask import Flask, render_template, jsonify, redirect, request
from utils.db import TracePreprocessor, DEFAULT_PRICING
from utils.viz import create_scatter_plot, create_task_success_heatmap, create_leaderboard, create_bar_chart
import plotly.utils
import json
from datetime import datetime

# List of contributors from creators.md
CONTRIBUTORS = [
    {"name": "Amit Arora", "affiliation": "Amazon"},
    {"name": "Aymeric Roucher", "affiliation": "Hugging Face"},
    {"name": "Hailey Schoelkopf", "affiliation": "Anthropic"},
    {"name": "Harsh Trivedi", "affiliation": "Stony Brook"},
    {"name": "Iason Gabriel", "affiliation": "Google DeepMind"},
    {"name": "Jelena Luketina", "affiliation": "UK AISI"},
    {"name": "JJ Allaire", "affiliation": "UK AISI"},
    {"name": "Laura Weidinger", "affiliation": "Google DeepMind"},
    {"name": "Madhur Prashant", "affiliation": "Amazon"},
    {"name": "Marius Hobbhahn", "affiliation": "Apollo Research"},
    {"name": "Maximillian Kaufmann", "affiliation": "UK AISI"},
    {"name": "Morgan McGuire", "affiliation": "Weights & Biases"},
    {"name": "Omar Khattab", "affiliation": "MIT"},
    {"name": "Parth Asawa", "affiliation": "UC Berkeley"},
    {"name": "Rishi Bommasani", "affiliation": "Stanford"},
    {"name": "Shreya Shankar", "affiliation": "UC Berkeley"},
    {"name": "Shayne Longpre", "affiliation": "MIT"},
    {"name": "William Isaac", "affiliation": "Google DeepMind"},
    {"name": "Yifan Mai", "affiliation": "Stanford"},
    {"name": "Zachary Siegel", "affiliation": "Princeton"}
]

def create_app():
    app = Flask(__name__, 
               static_folder='static',
               template_folder='templates')
    
    preprocessor = TracePreprocessor()
    
    @app.route('/')
    def index():
        total_agents = preprocessor.get_total_agents()
        total_benchmarks = preprocessor.get_total_benchmarks()
        return render_template('index.html', 
                             total_agents=total_agents, 
                             total_benchmarks=total_benchmarks,
                             contributors=CONTRIBUTORS)

    @app.route('/swebench')
    def swebench():
        # Get models used in SWE-bench benchmark
        swebench_models = preprocessor.get_models_for_benchmark('swebench_verified')
        
        # Filter pricing to only show models used in SWE-bench
        pricing = {model: DEFAULT_PRICING[model] for model in swebench_models if model in DEFAULT_PRICING}
        
        # Get data for SWE-bench
        results_df = preprocessor.get_parsed_results_with_costs('swebench_verified')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='swebench_verified')
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('swebench_verified'),
            'SWE-bench Verified'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'swebench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='swebench_verified'  # Add benchmark name for failure analysis
        )

    @app.route('/usaco')
    def usaco():
        # Get models used in USACO benchmark
        usaco_models = preprocessor.get_models_for_benchmark('usaco')
        
        # Filter pricing to only show models used in USACO
        pricing = {model: DEFAULT_PRICING[model] for model in usaco_models if model in DEFAULT_PRICING}
        
        # Get data for USACO
        results_df = preprocessor.get_parsed_results_with_costs('usaco')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='usaco')
        
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            preprocessor.get_parsed_results('usaco', aggregate=False),
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('usaco'),
            'USACO'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'usaco.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='usaco'  # Add benchmark name for failure analysis
        )

    @app.route('/update_pricing/<benchmark>', methods=['POST'])
    def update_pricing(benchmark):
        pricing = request.json
        
        if benchmark == 'agentharm':
            # Get updated results with new pricing for both benchmarks
            results_df = preprocessor.get_parsed_results_with_costs('agentharm', pricing)
            results_df_benign = preprocessor.get_parsed_results_with_costs('agentharm_benign', pricing)
            
            # Create updated leaderboards
            leaderboard_df = create_leaderboard(results_df, benchmark_name='agentharm')
            leaderboard_df_benign = create_leaderboard(results_df_benign, benchmark_name='agentharm_benign')
            
            # Create updated scatter plots
            scatter_plot = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            scatter_plot_benign = create_scatter_plot(
                results_df_benign,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            
            # Convert scatter plots to JSON
            scatter_plot_json = json.loads(json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder))
            scatter_plot_benign_json = json.loads(json.dumps(scatter_plot_benign, cls=plotly.utils.PlotlyJSONEncoder))
            
            return jsonify({
                'leaderboard': leaderboard_df.to_dict('records'),
                'leaderboard_benign': leaderboard_df_benign.to_dict('records'),
                'scatter_plot': scatter_plot_json,
                'scatter_plot_benign': scatter_plot_benign_json
            })
        else:
            # Get updated results with new pricing
            results_df = preprocessor.get_parsed_results_with_costs(benchmark, pricing)
                        
            # Create updated leaderboard
            leaderboard_df = create_leaderboard(results_df, benchmark_name=benchmark)
            
            # Create updated scatter plot
            scatter_plot = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            
            # Convert scatter plot to JSON
            scatter_plot_json = json.loads(json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder))
            
            return jsonify({
                'leaderboard': leaderboard_df.to_dict('records'),
                'scatter_plot': scatter_plot_json
            })

    @app.route('/appworld_test_normal')
    def appworld_test_normal():
        # Get models used in AppWorld benchmark
        appworld_models = preprocessor.get_models_for_benchmark('appworld_test_normal')
        
        # Filter pricing to only show models used in AppWorld
        pricing = {model: DEFAULT_PRICING[model] for model in appworld_models if model in DEFAULT_PRICING}
        
        # Get data for AppWorld
        results_df = preprocessor.get_parsed_results_with_costs('appworld_test_normal')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='appworld_test_normal')

        # Create scatter plot
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('appworld_test_normal'),
            'AppWorld'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'appworld_test_normal.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='appworld_test_normal'  # Add benchmark name for failure analysis
        )

    @app.route('/appworld_test_challenge')
    def appworld_test_challenge():
        # Get models used in AppWorld Challenge benchmark
        appworld_models = preprocessor.get_models_for_benchmark('appworld_test_challenge')
        pricing = {model: DEFAULT_PRICING[model] for model in appworld_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('appworld_test_challenge')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='appworld_test_challenge')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('appworld_test_challenge'),
            'AppWorld Challenge'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'appworld_test_challenge.html',  # Reuse the same template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='appworld_test_challenge'  # Add benchmark name for failure analysis
        )

    @app.route('/corebench_easy')
    def corebench_easy():
        corebench_models = preprocessor.get_models_for_benchmark('corebench_easy')
        pricing = {model: DEFAULT_PRICING[model] for model in corebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('corebench_easy')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='corebench_easy')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('corebench_easy'),
            'CORE-Bench-Easy'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'corebench.html',  # Will need to create this template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            difficulty="Easy",
            benchmark_name='corebench_easy'  # Add benchmark name for failure analysis
        )

    @app.route('/corebench_medium')
    def corebench_medium():
        corebench_models = preprocessor.get_models_for_benchmark('corebench_medium')
        pricing = {model: DEFAULT_PRICING[model] for model in corebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('corebench_medium')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='corebench_medium')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('corebench_medium'),
            'CORE-Bench-Medium'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'corebench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            difficulty="Medium",
            benchmark_name='corebench_medium'  # Add benchmark name for failure analysis
        )

    @app.route('/corebench_hard')
    def corebench_hard():
        corebench_models = preprocessor.get_models_for_benchmark('corebench_hard')
        pricing = {model: DEFAULT_PRICING[model] for model in corebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('corebench_hard')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='corebench_hard')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('corebench_hard'),
            'CORE-Bench-Hard'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'corebench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            difficulty="Hard",
            benchmark_name='corebench_hard'  # Add benchmark name for failure analysis
        )

    @app.route('/gaia')
    def gaia():
        gaia_models = preprocessor.get_models_for_benchmark('gaia')
        pricing = {model: DEFAULT_PRICING[model] for model in gaia_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('gaia')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='gaia')
                
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('gaia'),
            'GAIA'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'gaia.html',  # Will need to create this template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='gaia'  # Add benchmark name for failure analysis
        )

    @app.route('/cybench')
    def cybench():
        cybench_models = preprocessor.get_models_for_benchmark('cybench')
        pricing = {model: DEFAULT_PRICING[model] for model in cybench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('cybench')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='cybench')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('cybench'),
            'Cybench'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'cybench.html',  # Will need to create this template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='cybench'  # Add benchmark name for failure analysis
        )

    @app.route('/agentharm')
    def agentharm():
        # Get models and pricing for both benchmarks
        agentharm_models = preprocessor.get_models_for_benchmark('agentharm')
        agentharm_benign_models = preprocessor.get_models_for_benchmark('agentharm_benign')
        all_models = list(set(agentharm_models + agentharm_benign_models))
        pricing = {model: DEFAULT_PRICING[model] for model in all_models if model in DEFAULT_PRICING}
        
        # Get results for both benchmarks
        results_df = preprocessor.get_parsed_results_with_costs('agentharm')
        results_df_benign = preprocessor.get_parsed_results_with_costs('agentharm_benign')
        
        # Create leaderboards for both benchmarks
        leaderboard_df = create_leaderboard(results_df, benchmark_name='agentharm')
        leaderboard_df_benign = create_leaderboard(results_df_benign, benchmark_name='agentharm_benign')
        
        # Create scatter plots for both benchmarks
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_benign = create_scatter_plot(
            results_df_benign,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        scatter_plot_benign_json = json.dumps(scatter_plot_benign, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmaps for both benchmarks
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('agentharm'),
            'AgentHarm'
        )
        heatmap_benign = create_task_success_heatmap(
            preprocessor.get_task_success_data('agentharm_benign'),
            'AgentHarm Benign'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        heatmap_benign_json = json.dumps(heatmap_benign, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'agentharm.html',
            leaderboard=leaderboard_df.to_dict('records'),
            leaderboard_benign=leaderboard_df_benign.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_benign=scatter_plot_benign_json,
            heatmap=heatmap_json,
            heatmap_benign=heatmap_benign_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='agentharm'
        )

    @app.route('/taubench_retail')
    def taubench_retail():
        # Get models used in TAU-bench Retail benchmark
        taubench_retail_models = preprocessor.get_models_for_benchmark('taubench_retail')
        pricing = {model: DEFAULT_PRICING[model] for model in taubench_retail_models if model in DEFAULT_PRICING}
        
        # Get data for TAU-bench Retail
        results_df = preprocessor.get_parsed_results_with_costs('taubench_retail')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='taubench_retail')
        
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('taubench_retail'),
            'TAU-bench Retail'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'taubench_retail.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='taubench_retail'
        )

    @app.route('/taubench_airline')
    def taubench_airline():
        # Get models used in TAU-bench Airline benchmark
        taubench_airline_models = preprocessor.get_models_for_benchmark('taubench_airline')
        pricing = {model: DEFAULT_PRICING[model] for model in taubench_airline_models if model in DEFAULT_PRICING}
        
        # Get data for TAU-bench Airline
        results_df = preprocessor.get_parsed_results_with_costs('taubench_airline')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='taubench_airline')
        
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('taubench_airline'),
            'TAU-bench Airline'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'taubench_airline.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='taubench_airline'
        )

    @app.route('/swebench_verified_mini')
    def swebench_verified_mini():
        swebench_models = preprocessor.get_models_for_benchmark('swebench_verified_mini')
        pricing = {model: DEFAULT_PRICING[model] for model in swebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('swebench_verified_mini')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='swebench_verified_mini')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('swebench_verified_mini'),
            'SWE-bench Verified (Mini)'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'swebench_verified_mini.html',  # Use the new template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='swebench_verified_mini'  # Add benchmark name for failure analysis
        )

    @app.route('/failure_report/<benchmark>')
    def failure_report(benchmark):
        agent_name = request.args.get('agent')
        if not agent_name:
            return jsonify({'error': 'Agent name is required'}), 400
            
        # Get failure report for the agent
        failure_report = preprocessor.get_failure_report(agent_name, benchmark)
        if not failure_report:
            return jsonify({
                'failure_categories': [],
                'chart_data': None
            })
            
        # Create bar chart data
        categories = []
        counts = []
        for category in failure_report['failure_categories']:
            categories.append(category['category_name'])
            # Count tasks in this category
            count = sum(1 for _, classification in failure_report['task_classifications'].items() 
                       if classification['category_id'] == str(len(categories)))
            counts.append(count)
            
        chart = create_bar_chart(
            categories,
            counts,
            "Number of Tasks",
            "Failure Categories",
            "Distribution of Failure Categories"
        )
        
        return jsonify({
            'failure_categories': failure_report['failure_categories'],
            'chart_data': json.loads(json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder))
        })

    @app.route('/available_agents/<benchmark>')
    def available_agents(benchmark):
        # Get all agents for the benchmark
        agents = preprocessor.get_all_agents(benchmark)
        
        # Filter to only agents with failure reports
        agents_with_reports = [
            agent for agent in agents 
            if preprocessor.get_failure_report(agent, benchmark) is not None
        ]
        
        return jsonify(agents_with_reports)

    @app.route('/creators')
    def creators():
        return render_template('creators.html', contributors=CONTRIBUTORS)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')