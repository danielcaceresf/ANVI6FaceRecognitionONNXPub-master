﻿#pragma checksum "C:\Lab_Api_Face\ANVI6FaceRecognitionONNXPub-master\App4\MainPage.xaml" "{406ea660-64cf-4c82-b6f0-42d48172a799}" "A8E3CBC6C754402D723297CC64586169"
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace App4
{
    partial class MainPage : 
        global::Windows.UI.Xaml.Controls.Page, 
        global::Windows.UI.Xaml.Markup.IComponentConnector,
        global::Windows.UI.Xaml.Markup.IComponentConnector2
    {
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        private static class XamlBindingSetters
        {
            public static void Set_Windows_UI_Xaml_Controls_ItemsControl_ItemsSource(global::Windows.UI.Xaml.Controls.ItemsControl obj, global::System.Object value, string targetNullValue)
            {
                if (value == null && targetNullValue != null)
                {
                    value = (global::System.Object) global::Windows.UI.Xaml.Markup.XamlBindingHelper.ConvertValue(typeof(global::System.Object), targetNullValue);
                }
                obj.ItemsSource = value;
            }
        };

        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        private class MainPage_obj1_Bindings :
            global::Windows.UI.Xaml.Markup.IDataTemplateComponent,
            global::Windows.UI.Xaml.Markup.IXamlBindScopeDiagnostics,
            global::Windows.UI.Xaml.Markup.IComponentConnector,
            IMainPage_Bindings
        {
            private global::App4.MainPage dataRoot;
            private bool initialized = false;
            private const int NOT_PHASED = (1 << 31);
            private const int DATA_CHANGED = (1 << 30);

            // Fields for each control that has bindings.
            private global::Windows.UI.Xaml.Controls.ListBox obj15;

            // Static fields for each binding's enabled/disabled state
            private static bool isobj15ItemsSourceDisabled = false;

            public MainPage_obj1_Bindings()
            {
            }

            public void Disable(int lineNumber, int columnNumber)
            {
                if (lineNumber == 63 && columnNumber == 170)
                {
                    isobj15ItemsSourceDisabled = true;
                }
            }

            // IComponentConnector

            public void Connect(int connectionId, global::System.Object target)
            {
                switch(connectionId)
                {
                    case 15: // MainPage.xaml line 63
                        this.obj15 = (global::Windows.UI.Xaml.Controls.ListBox)target;
                        break;
                    default:
                        break;
                }
            }

            // IDataTemplateComponent

            public void ProcessBindings(global::System.Object item, int itemIndex, int phase, out int nextPhase)
            {
                throw new global::System.NotImplementedException();
            }

            public void Recycle()
            {
                throw new global::System.NotImplementedException();
            }

            // IMainPage_Bindings

            public void Initialize()
            {
                if (!this.initialized)
                {
                    this.Update();
                }
            }
            
            public void Update()
            {
                this.Update_(this.dataRoot, NOT_PHASED);
                this.initialized = true;
            }

            public void StopTracking()
            {
            }

            public void DisconnectUnloadedObject(int connectionId)
            {
                throw new global::System.ArgumentException("No unloadable elements to disconnect.");
            }

            public bool SetDataRoot(global::System.Object newDataRoot)
            {
                if (newDataRoot != null)
                {
                    this.dataRoot = (global::App4.MainPage)newDataRoot;
                    return true;
                }
                return false;
            }

            public void Loading(global::Windows.UI.Xaml.FrameworkElement src, object data)
            {
                this.Initialize();
            }

            // Update methods for each path node used in binding steps.
            private void Update_(global::App4.MainPage obj, int phase)
            {
                if (obj != null)
                {
                    if ((phase & (NOT_PHASED | (1 << 0))) != 0)
                    {
                        this.Update_learningModelDeviceKinds(obj.learningModelDeviceKinds, phase);
                    }
                }
            }
            private void Update_learningModelDeviceKinds(global::System.Collections.ObjectModel.ObservableCollection<global::Windows.AI.MachineLearning.LearningModelDeviceKind> obj, int phase)
            {
                if ((phase & ((1 << 0) | NOT_PHASED )) != 0)
                {
                    // MainPage.xaml line 63
                    if (!isobj15ItemsSourceDisabled)
                    {
                        XamlBindingSetters.Set_Windows_UI_Xaml_Controls_ItemsControl_ItemsSource(this.obj15, obj, null);
                    }
                }
            }
        }
        /// <summary>
        /// Connect()
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public void Connect(int connectionId, object target)
        {
            switch(connectionId)
            {
            case 2: // MainPage.xaml line 93
                {
                    this.listaCaras = (global::Windows.UI.Xaml.Controls.ListView)(target);
                }
                break;
            case 3: // MainPage.xaml line 100
                {
                    this.imagenCamaraWeb = (global::Windows.UI.Xaml.Controls.Image)(target);
                }
                break;
            case 4: // MainPage.xaml line 102
                {
                    global::Windows.UI.Xaml.Controls.Button element4 = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)element4).Click += this.VersionVideo_Click;
                }
                break;
            case 6: // MainPage.xaml line 68
                {
                    this.CamPreview = (global::Windows.UI.Xaml.Controls.CaptureElement)(target);
                }
                break;
            case 7: // MainPage.xaml line 70
                {
                    this.VisualizationCanvas = (global::Windows.UI.Xaml.Controls.Canvas)(target);
                    ((global::Windows.UI.Xaml.Controls.Canvas)this.VisualizationCanvas).PointerPressed += this.VisualizationCanvas_PointerPressed;
                }
                break;
            case 8: // MainPage.xaml line 75
                {
                    this.RectIdentificando = (global::Windows.UI.Xaml.Shapes.Rectangle)(target);
                }
                break;
            case 9: // MainPage.xaml line 76
                {
                    this.txtIdentificando = (global::Windows.UI.Xaml.Controls.TextBlock)(target);
                }
                break;
            case 10: // MainPage.xaml line 77
                {
                    this.txtResult = (global::Windows.UI.Xaml.Controls.TextBlock)(target);
                }
                break;
            case 11: // MainPage.xaml line 78
                {
                    this.txtResultServicio = (global::Windows.UI.Xaml.Controls.TextBlock)(target);
                }
                break;
            case 12: // MainPage.xaml line 79
                {
                    this.mediaPlayerUWP = (global::Windows.UI.Xaml.Controls.MediaElement)(target);
                }
                break;
            case 13: // MainPage.xaml line 81
                {
                    this.previewFoto = (global::Windows.UI.Xaml.Controls.Image)(target);
                }
                break;
            case 14: // MainPage.xaml line 62
                {
                    this.lstBoxCamaras = (global::Windows.UI.Xaml.Controls.ListBox)(target);
                    ((global::Windows.UI.Xaml.Controls.ListBox)this.lstBoxCamaras).SelectionChanged += this.lstBoxCamaras_SelectionChanged;
                }
                break;
            case 15: // MainPage.xaml line 63
                {
                    this.lstBoxModelo = (global::Windows.UI.Xaml.Controls.ListBox)(target);
                    ((global::Windows.UI.Xaml.Controls.ListBox)this.lstBoxModelo).SelectionChanged += this.lstBoxModelo_SelectionChanged;
                }
                break;
            case 16: // MainPage.xaml line 64
                {
                    this.btnTomarFoto = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.btnTomarFoto).Click += this.btnTomarFoto_Click;
                }
                break;
            case 17: // MainPage.xaml line 65
                {
                    this.btnIniciarStream = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.btnIniciarStream).Click += this.CameraStreamingButton_Click;
                }
                break;
            default:
                break;
            }
            this._contentLoaded = true;
        }

        /// <summary>
        /// GetBindingConnector(int connectionId, object target)
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public global::Windows.UI.Xaml.Markup.IComponentConnector GetBindingConnector(int connectionId, object target)
        {
            global::Windows.UI.Xaml.Markup.IComponentConnector returnValue = null;
            switch(connectionId)
            {
            case 1: // MainPage.xaml line 1
                {                    
                    global::Windows.UI.Xaml.Controls.Page element1 = (global::Windows.UI.Xaml.Controls.Page)target;
                    MainPage_obj1_Bindings bindings = new MainPage_obj1_Bindings();
                    returnValue = bindings;
                    bindings.SetDataRoot(this);
                    this.Bindings = bindings;
                    element1.Loading += bindings.Loading;
                    global::Windows.UI.Xaml.Markup.XamlBindingHelper.SetDataTemplateComponent(element1, bindings);
                }
                break;
            }
            return returnValue;
        }
    }
}

