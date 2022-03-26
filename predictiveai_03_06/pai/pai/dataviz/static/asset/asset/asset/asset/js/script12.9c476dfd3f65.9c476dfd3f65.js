

(function() {

    var width, height, largeHeader, canvas, ctx, points, target, animateHeader = true;

    // Main
    initHeader();
    initAnimation();
    addListeners();

    function initHeader() {
        width = window.innerWidth;
        height = window.innerHeight;
        target = {x: width/2, y: height/2};

        largeHeader = document.getElementById('large-header');
        largeHeader.style.height = height+'px';

        canvas = document.getElementById('demo-canvas');
        canvas.width = width;
        canvas.height = height;
        ctx = canvas.getContext('2d');

        // create points
        points = [];
        for(var x = 0; x < width; x = x + width/30) {
            for(var y = 0; y < height; y = y + height/15) {
                var px = x + Math.random()*width/5;
                var py = y + Math.random()*height/5;
                var p = {x: px, originX: px, y: py, originY: py };
                points.push(p);
            }
        }

        // for each point find the 5 closest points
        for(var i = 0; i < points.length; i++) {
            var closest = [];
            var p1 = points[i];
            for(var j = 0; j < points.length; j++) {
                var p2 = points[j]
                if(!(p1 == p2)) {
                    var placed = false;
                    for(var k = 0; k < 5; k++) {
                        if(!placed) {
                            if(closest[k] == undefined) {
                                closest[k] = p2;
                                placed = true;
                            }
                        }
                    }

                    for(var k = 0; k < 5; k++) {
                        if(!placed) {
                            if(getDistance(p1, p2) < getDistance(p1, closest[k])) {
                                closest[k] = p2;
                                placed = true;
                            }
                        }
                    }
                }
            }
            p1.closest = closest;
        }

        // assign a circle to each point
        for(var i in points) {
            var c = new Circle(points[i], 2+Math.random()*2, 'rgba(255,255,255,0.3)');
            points[i].circle = c;
        }
    }

    // Event handling
    function addListeners() {
        if(!('ontouchstart' in window)) {
            window.addEventListener('mousemove', mouseMove);
        }
        window.addEventListener('scroll', scrollCheck);
        window.addEventListener('resize', resize);
    }

    function mouseMove(e) {
        var posx = posy = 0;
        if (e.pageX || e.pageY) {
            posx = e.pageX;
            posy = e.pageY;
        }
        else if (e.clientX || e.clientY)    {
            posx = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
            posy = e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
        }
        target.x = posx;
        target.y = posy;
    }

    function scrollCheck() {
        if(document.body.scrollTop > height) animateHeader = false;
        else animateHeader = true;
    }

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        largeHeader.style.height = height+'px';
        canvas.width = width;
        canvas.height = height;
    }

    // animation
    function initAnimation() {
        animate();
        for(var i in points) {
            shiftPoint(points[i]);
        }
    }

    function animate() {
        if(animateHeader) {
            ctx.clearRect(0,0,width,height);
            for(var i in points) {
                // detect points in range
                if(Math.abs(getDistance(target, points[i])) < 4000) {
                    points[i].active = 0.3;
                    points[i].circle.active = 0.6;
                } else if(Math.abs(getDistance(target, points[i])) < 20000) {
                    points[i].active = 0.1;
                    points[i].circle.active = 0.3;
                } else if(Math.abs(getDistance(target, points[i])) < 40000) {
                    points[i].active = 0.02;
                    points[i].circle.active = 0.1;
                } else {
                    points[i].active = 0;
                    points[i].circle.active = 0;
                }

                drawLines(points[i]);
                points[i].circle.draw();
            }
        }
        requestAnimationFrame(animate);
    }

    function shiftPoint(p) {
        TweenLite.to(p, 1+1*Math.random(), {x:p.originX-50+Math.random()*100,
            y: p.originY-50+Math.random()*100, ease:Circ.easeInOut,
            onComplete: function() {
                shiftPoint(p);
            }});
    }

    // Canvas manipulation
    function drawLines(p) {
        if(!p.active) return;
        for(var i in p.closest) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.closest[i].x, p.closest[i].y);
            ctx.strokeStyle = 'rgba(156,217,249,'+ p.active+')';
            ctx.stroke();
        }
    }

    function Circle(pos,rad,color) {
        var _this = this;

        // constructor
        (function() {
            _this.pos = pos || null;
            _this.radius = rad || null;
            _this.color = color || null;
        })();

        this.draw = function() {
            if(!_this.active) return;
            ctx.beginPath();
            ctx.arc(_this.pos.x, _this.pos.y, _this.radius, 0, 2 * Math.PI, false);
            ctx.fillStyle = 'rgba(156,217,249,'+ _this.active+')';
            ctx.fill();
        };
    }

    // Util
    function getDistance(p1, p2) {
        return Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2);
    }

})();


//nav bar
toggle = document.querySelectorAll(".toggle")[0];
nav = document.querySelectorAll("nav")[0];
toggle_open_text = 'Menu';
toggle_close_text = 'Close';

toggle.addEventListener('click', function() {
	nav.classList.toggle('open');

  if (nav.classList.contains('open')) {
    toggle.innerHTML = toggle_close_text;
  } else {
    toggle.innerHTML = toggle_open_text;
  }
}, false);

setTimeout(function(){
	nav.classList.toggle('open');
}, 800);



//view js
var myVar;

function myFunction11() {
  myVar = setTimeout(showPage,3000);
}

function showPage() {
  document.getElementById("loader11").style.display = "none";
  document.getElementById("myDiv11").style.display = "block";
}
//vies js end
// Created by @conmarap.

//multiselect
$(document).ready(function() {
        $('#multiple-checkboxes').multiselect({
          includeSelectAllOption: true,
        });
    });

//visz

function myFunction() {
   var x = document.getElementById("myText").value;
    console.log(x)

    var a10 = document.getElementById("myDIV10");
    var a11 = document.getElementById("myDIV11");
    var a12 = document.getElementById("myDIV12");
    var a13 = document.getElementById("myDIV13");
    var a14 = document.getElementById("myDIV14");
    var a15 = document.getElementById("myDIV15");
    var a16 = document.getElementById("myDIV16");
    var a17 = document.getElementById("myDIV17");
    var a18 = document.getElementById("myDIV18");
    var a19 = document.getElementById("myDIV19");
    var a20 = document.getElementById("myDIV20");
     var a21 = document.getElementById("myDIV21");
     var a22 = document.getElementById("myDIV22");
     var a23 = document.getElementById("myDIV23");

    if(x == "BarChart")
        {

         a10.style.display= "block";



             for (i = 10; i <24 ; i++){
                 if (i === 10) { continue; }
                 var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }

        }
    if(x == "ScatterCategorical")
        {


            a11.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 11) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "lm-Plot")
        {


            a12.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 12) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "Violin-Plot")
        {


            a13.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 13) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "Count-Plot")
        {


            a14.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 14) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "Histogram")
        {


            a15.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 15) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "Distibution-Plot")
        {


            a16.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 16) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "Line-Plot")
        {


            a17.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 17) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "Heat-map")
        {


            a18.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 18) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "Pie-chart")
        {


            a19.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 19) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x == "3D-Scatter")
        {


            a20.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 20) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";
             }
        }
    if(x =="ScatterNumerical" )
        {
             a21.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 21) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";

             }


        }
     if(x =="Box-Plot" )
        {
             a22.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 22) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";

             }


        }
 if(x =="Pair-Plot" )
        {
             a23.style.display= "block";

             var s="myDIV";
             for (i = 10; i <24 ; i++){
                 if (i === 23) { continue; }
                var p='a'+i;
                 p = document.getElementById("myDIV"+i);


                 p.style.display= "none";

             }


        }
}

